import torch
from typing import Optional
from transformers import Trainer, TrainerCallback

from .models import Florence2Processor
from .utils import (print_trainable_parameters, check_pretrained_load, 
                    rank0_print, maybe_zero3_gathered_parameters, 
                    freeze_params, unfreeze_params, freeze_partial_embeddings)


def initialize_model(model_args, training_args):
    model_kwargs = {}
    if model_args.model_type == "florence2":
        from .models import Florence2ForConditionalGeneration

        if model_args.model_revision is None:
            from .models import default_version_commit_id
            if "base" in model_args.model_name_or_path.lower():
                model_args.model_revision = default_version_commit_id["florence2"]["base"]
            elif "large" in model_args.model_name_or_path.lower():
                model_args.model_revision = default_version_commit_id["florence2"]["large"]
            else:
                raise ValueError("model_revision must be specified for custom model.")
        
        MODEL_CLASS = Florence2ForConditionalGeneration
    elif model_args.model_type == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration

        if model_args.model_revision is None:
            model_args.model_revision = "main"

        MODEL_CLASS = Qwen2VLForConditionalGeneration
    elif model_args.model_type in ["internvl2", "internvl_chat"]:
        from .models import InternVLChatModel

        if model_args.model_revision is None:
            model_args.model_revision = "main"
    
        MODEL_CLASS = InternVLChatModel
    elif model_args.model_type == "llava_qwen":
        from .models import LlavaQwenForCausalLM

        if model_args.model_revision is None:
            model_args.model_revision = "main"
    
        MODEL_CLASS = LlavaQwenForCausalLM
    else:
        raise ValueError(f"model_type={model_args.model_type} is not supported.")
    
    model = MODEL_CLASS.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        low_cpu_mem_usage=True,
        revision=model_args.model_revision,
        **model_kwargs
    )
    
    rank0_print(check_pretrained_load(model, model_args.model_name_or_path))

    if model_args.model_type == "florence2" and not hasattr(model.config, "hidden_size"):
        # deepspeed may touch this attribute
        model.config.hidden_size = model.config.text_config.d_model

    if hasattr(model_args, "response_format"):
        model.response_format = model_args.response_format
        model.config.response_format = model_args.response_format

    if model_args.language_model_lora not in [0, None]:
        assert model_args.model_type == "internvl2", \
            "LoRA is only supported for InternVL2 model now."
        rank0_print("Adding LoRA adapters...")
        model.wrap_llm_lora(r=model_args.language_model_lora, lora_alpha=2 * model_args.language_model_lora)
        model.config.use_llm_lora = model_args.language_model_lora
    return model
        

def setting_training_options(model, training_args):
    model_cls_name = type(model).__name__
    assert model_cls_name in [
        "Florence2ForConditionalGeneration", "Qwen2VLForConditionalGeneration", 
        "InternVLChatModel", "LlavaQwenForCausalLM"
    ], f"Model class name {model_cls_name} is not supported."
        
    if training_args.freeze_vision:
        if model_cls_name in ["Florence2ForConditionalGeneration", "InternVLChatModel"]:
            freeze_params(model.vision_model)
        elif model_cls_name == "LlavaQwenForCausalLM":
            freeze_params(model.get_vision_tower())
        elif model_cls_name == "Qwen2VLForConditionalGeneration":
            freeze_params([model.visual.patch_embed, model.visual.rotary_pos_emb, model.visual.blocks])
    
    if training_args.freeze_language:
        if model_cls_name in ["Florence2ForConditionalGeneration", "InternVLChatModel"]:
            freeze_params(model.language_model)
        elif model_cls_name == "LlavaQwenForCausalLM":
            freeze_params([model.get_model().layers, model.get_input_embeddings(), model.get_output_embeddings()])
        elif model_cls_name == "Qwen2VLForConditionalGeneration":
            freeze_params(model.model)

    if training_args.freeze_multimodal_projection:
        if model_cls_name == "Florence2ForConditionalGeneration":
            freeze_params([model.image_projection, model.image_proj_norm, model.image_pos_embed, model.visual_temporal_embed])
        elif model_cls_name == "Qwen2VLForConditionalGeneration":
            freeze_params(model.visual.merger)
        elif model_cls_name == "InternVLChatModel":
            freeze_params(model.mlp1)
            
    print_trainable_parameters(model)

    # hack for gradient checkpointing
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model
        

def resize_positional_embeddings(model, model_args):
    if model_args.image_square_length is None and model_args.language_model_max_length is None:
        return model, {}
    
    if model_args.model_type != "florence2":
        rank0_print(
            "The resize_positional_embeddings function is only used for Florence2 model. "
            f"The image_square_length option is not required for {model_args.model_type} because of dynamic input resolutions."
            "The language_model_max_length option will only be used in dataset."
        )
        return model, {}

    config = model.config
    processor_kwargs = {}
    
    if model_args.image_square_length is not None:
        size = {"height": model_args.image_square_length, "width": model_args.image_square_length}
        feature_square_length = int(model_args.image_square_length / 32)
        assert feature_square_length * 32 == model_args.image_square_length, \
            "image_square_length must be a multiple of 32"
        
        image_seq_length = feature_square_length ** 2 + 1
        
        vision_pos_embed_square_length = config.vision_config.image_pos_embed["max_pos_embeddings"]
        if vision_pos_embed_square_length < feature_square_length:
            config.vision_config.image_pos_embed["max_pos_embeddings"] = feature_square_length
            
            def _resize_vision_pos_embed(module, size):
                with maybe_zero3_gathered_parameters(module.weight, modifier_rank=0):
                    pe_old = module.weight.data
                    pe_new = torch.nn.functional.interpolate(
                        pe_old.permute(1, 0).unsqueeze(0), 
                        size=size, mode='linear', align_corners=True).squeeze(0).permute(1, 0)
                new_embedding = type(module).from_pretrained(pe_new, freeze=False)  # nn.Embeddings
                return new_embedding
                
            model.image_pos_embed.row_embeddings = _resize_vision_pos_embed(
                model.image_pos_embed.row_embeddings, feature_square_length)
            model.image_pos_embed.column_embeddings = _resize_vision_pos_embed(
                model.image_pos_embed.column_embeddings, feature_square_length)
        
        processor_kwargs.update({
            "size": size, "crop_size": size,
            "image_seq_length": image_seq_length
        })
            
    if model_args.language_model_max_length is not None:
        config.text_config.max_position_embeddings = model_args.language_model_max_length
        
        def _resize_language_pos_embed(module, size):
            with maybe_zero3_gathered_parameters(module.weight, modifier_rank=0):
                pe_old = module.weight.data
                old_size, embed_dim = pe_old.shape
                size += module.offset
                # print(f"size={size}, pe_old.shape={pe_old.shape}, module.offset={module.offset}")
            if size - old_size <= 0:
                return module
            with maybe_zero3_gathered_parameters(module.weight, modifier_rank=0):
                pe_to_cat = torch.randn(size - old_size, embed_dim, device=pe_old.device, dtype=pe_old.dtype) * 0.02
                pe_new = torch.cat([pe_old, pe_to_cat], dim=0)
                new_embedding = type(module).from_pretrained(pe_new, freeze=False)  # Florence2LearnedPositionalEmbedding
        
            def _freeze_old_embeddings(grad, _old_size=old_size):
                grad[:_old_size].zero_()
                return grad
            
            new_embedding.weight.register_hook(_freeze_old_embeddings)
            return new_embedding
        
        model.language_model.model.encoder.embed_positions = _resize_language_pos_embed(
            model.language_model.model.encoder.embed_positions, model_args.language_model_max_length)
        model.language_model.model.decoder.embed_positions = _resize_language_pos_embed(
            model.language_model.model.decoder.embed_positions, model_args.language_model_max_length)
    return model, processor_kwargs


def initialize_processor(model_args, **processor_kwargs):
    if model_args.model_type == "florence2":
        processor = Florence2Processor.from_pretrained(model_args.model_name_or_path, **processor_kwargs)
    elif model_args.model_type == "qwen2_vl":
        from transformers import Qwen2VLProcessor
        processor = Qwen2VLProcessor.from_pretrained(model_args.model_name_or_path, **processor_kwargs)
    elif model_args.model_type in ["internvl2", "internvl_chat"]:
        from .models import SimpleInternVL2Processor
        processor = SimpleInternVL2Processor.from_pretrained(model_args.model_name_or_path, **processor_kwargs)
    elif model_args.model_type == "llava_qwen":
        from .models import SimpleLlavaQwenProcessor
        processor = SimpleLlavaQwenProcessor.from_pretrained(model_args.model_name_or_path, **processor_kwargs)
    else:
        raise ValueError(f"model_type={model_args.model_type} is not supported.")
    return processor


def _init_coordinate_tokens_weight(model, num_new_tokens):
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

    input_embeddings[-num_new_tokens:] = input_embeddings_avg
    output_embeddings[-num_new_tokens:] = output_embeddings_avg


def add_coordinate_tokens(model, processor, NUM_BBOX_BINS=1000):
    if model.config.model_type == "florence2":
        return model, processor

    tokenizer = processor.tokenizer
    model_class = type(model).__name__
    
    old_tokenizer_len = len(tokenizer)
    old_embedding_len = len(model.get_input_embeddings().weight.data)

    coordinate_tokens = [f"<loc_{i}>" for i in range(NUM_BBOX_BINS)] + ["<sep>"]
    num_new_tokens = processor.tokenizer.add_tokens(coordinate_tokens, special_tokens=True)
    if num_new_tokens > 0 or old_tokenizer_len != old_embedding_len:
        # The embedding number of Qwen model is more than token number of tokenizer
        rank0_print(f"Added {num_new_tokens} new tokens to the tokenizer. Hence, resizing the "
                    f"language model's output embeddings. ({old_tokenizer_len}->{len(tokenizer)})")

        if model_class in ["Qwen2VLForConditionalGeneration", "LlavaQwenForCausalLM"]:
            model.config.tie_word_embeddings = False
            model.model.resize_token_embeddings(len(tokenizer))
            
            _init_coordinate_tokens_weight(model, num_new_tokens)

            model.config.vocab_size = len(tokenizer)
            model.model.config.vocab_size = len(tokenizer)
        elif model_class == "InternVLChatModel":
            model.config.tie_word_embeddings = False
            model.resize_token_embeddings(len(tokenizer))
            
            _init_coordinate_tokens_weight(model, num_new_tokens)

            model.config.vocab_size = len(tokenizer)
            model.language_model.config.vocab_size = len(tokenizer)
        else:
            raise ValueError(f"model_class={model_class} is not supported.")

        # check whether the lora is initialized for the model.
        # if lora is initialized, we should make sure the coordinate tokens are trainable.
        
        language_model_lora = getattr(model.config, "use_llm_lora", None)  # maybe only used for InternVL2 model
        if language_model_lora is not None:
            if language_model_lora > 0:
                # plan A: just unfreeze the whole coordinate tokens
                unfreeze_params(model.get_input_embeddings())
                unfreeze_params(model.get_output_embeddings())

                # # plan B:  we should test freezing the previous tokens and only tune the new tokens, that's interesting
                # freeze_partial_embeddings(model.get_output_embeddings(), old_tokenizer_len)
                # freeze_partial_embeddings(model.get_input_embeddings(), old_tokenizer_len)
    return model, processor


def deal_with_response_format(model, processor):
    if model.response_format == "florence2":
        model, processor = add_coordinate_tokens(model, processor)
    elif model.response_format == "allseeing":
        assert model.config.model_type != "florence2", \
            "The allseeing response format does not support for florence2 model."
    else:
        raise NotImplementedError(f"response_format={model.response_format} is not supported.")
    return model, processor


def initialize_model_and_processor(model_args, training_args):
    model = initialize_model(model_args, training_args)
    model = setting_training_options(model, training_args)
    model, processor_kwargs = resize_positional_embeddings(model, model_args)
    processor = initialize_processor(model_args, **processor_kwargs)
    model, processor = deal_with_response_format(model, processor)
    if type(model).__name__ == "InternVLChatModel":
        model.img_context_token_id = processor.img_context_token_id
    return model, processor


class DatasetResetCallback(TrainerCallback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        seed = int(state.epoch) if state.epoch is not None else 0
        self.dataset.deal_with_multidataset_mode(seed)


class CustomTrainer(Trainer):
    def __init__(self, *args, processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir, state_dict)
        
        if getattr(self, "processor", None) is not None:
            self.processor.save_pretrained(output_dir)

    def save_final_model(self):
        self.save_state()
        if self.deepspeed:
            torch.cuda.synchronize()
            self.save_model(self.args.output_dir)
        else:
            state_dict = self.model.state_dict()
            if self.args.should_save:
                cpu_state_dict = {
                    key: value.cpu()
                    for key, value in state_dict.items()
                }
                del state_dict
                self._save(self.args.output_dir, state_dict=cpu_state_dict)
