import os
import json
import requests
from collections import namedtuple

from PIL import Image
import torch
from torch import nn

from accelerate.utils import send_to_device
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration, Qwen2VLProcessor, AutoConfig
from . import (
    Florence2Config, Florence2Processor, Florence2ForConditionalGeneration, 
    InternVLChatConfig, InternVLChatModel, SimpleInternVL2Processor, 
    LlavaQwenForCausalLM, LlavaQwenConfig, SimpleLlavaQwenProcessor
)
from .internvl2.allseeing_format import extract_objects
from ..utils import get_num_parameters, get_torch_dtype


class BaseInferencer():

    model_type = None
    CONFIG_CLASS = None
    MODEL_CLASS = None
    PROCESSOR_CLASS = None

    def __init__(self, 
                 model_ckpt_path, 
                 device, 
                 torch_dtype=None, 
                 attn_implementation="flash_attention_2",
                 wrap_with_ddp=False, 
                 task_prompt='<ROD>', 
                 max_max_length=8192):
        self.model_ckpt_path = model_ckpt_path
        self.device = device
        self.torch_dtype = get_torch_dtype(torch_dtype)
        self.attn_implementation = attn_implementation
        self.wrap_with_ddp = wrap_with_ddp
        self.task_prompt = task_prompt
        self.max_max_length = max_max_length
        self.model_loaded = False
        self.load_processor()

    def load_model(self):
        if self.torch_dtype is None:
            model_config = self.CONFIG_CLASS.from_pretrained(self.model_ckpt_path)
            self.torch_dtype = get_torch_dtype(model_config.torch_dtype)
        self.model = self.MODEL_CLASS.from_pretrained(
            self.model_ckpt_path, 
            attn_implementation=self.attn_implementation, 
            torch_dtype=self.torch_dtype,
        ).eval().to(self.device)
        if self.wrap_with_ddp:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device])
        self.model_loaded = True
        self.print_model_info()

        if self.model_type == "florence2":
            self.max_length = self.model_config.text_config.max_position_embeddings
        elif self.model_type in ["internvl2", "internvl_chat"]:
            self.model.img_context_token_id = self.processor.img_context_token_id
            self.max_length = self.model_config.llm_config.max_position_embeddings
        elif self.model_type == "qwen2_vl":
            self.max_length = self.model_config.max_position_embeddings
        elif self.model_type == "llava_qwen":
            self.max_length = getattr(
                self.model_config, 
                "max_position_embeddings", 
                self.model_config.max_position_embeddings)
        else:
            self.max_length = 32768
        self.max_length = min(self.max_length, self.max_max_length)

    def load_processor(self):
        self.load_model_processor()
        if self.PROCESSOR_CLASS != Florence2Processor:
            self.load_florence2_processor()
        else:
            self.florence2_processor = self.processor

    def load_model_processor(self):
        if self.model_type == "florence2" and \
            not os.path.exists(os.path.join(self.model_ckpt_path, "preprocessor_config.json")) and \
            os.path.exists(self.model_ckpt_path) and \
            os.path.exists(os.path.join(self.model_ckpt_path, "config.json")):
            with open(os.path.join(self.model_ckpt_path, "config.json"), "r") as f:
                pretrained_name_or_path = json.load(f)["_name_or_path"]
            print(f"Loading processor from {pretrained_name_or_path}, NOTE that you seems forget to call processor.save_pretrained()")
        else:
            pretrained_name_or_path = self.model_ckpt_path

        self.processor = self.PROCESSOR_CLASS.from_pretrained(pretrained_name_or_path)


    def load_florence2_processor(self):
        self.florence2_processor = Florence2Processor.from_pretrained("microsoft/Florence-2-large", tokenizer=self.processor.tokenizer)
        # if not hasattr(self.processor, "image_processor"):
        #     dummy_image_processor = namedtuple("image_processor", ["image_seq_length"])("")
        # else:
        #     self.processor.image_processor.image_seq_length = ""
        #     dummy_image_processor = self.processor.image_processor
        # self.florence2_processor = Florence2Processor(image_processor=dummy_image_processor, tokenizer=self.processor.tokenizer)
        # self.florence2_processor.tasks_answer_post_processing_type = Florence2Processor.tasks_answer_post_processing_type
        # self.florence2_processor.task_prompts_without_inputs = Florence2Processor.task_prompts_without_inputs

    def data_to_dtype(self, data_dict):
        for key, value in data_dict.items():
            if (isinstance(value, torch.Tensor) and torch.is_floating_point(value)) \
                or key in ("images", "image", "pixel_values"):
                data_dict[key] = value.to(self.torch_dtype)
        return data_dict
    
    def __call__(self, image, text_input=None):
        if not self.model_loaded:
            self.load_model()

        if os.path.exists(image):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, str):
            image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
        
        if text_input is None:
            prompt = self.task_prompt
        else:
            prompt = self.task_prompt + text_input

        if self.model_type != "florence2":
            prompt = self.florence2_processor._construct_prompts([prompt])[0]
            
        if self.model_type == "qwen2_vl":
            prompt = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            ]
            prompt = self.processor.apply_chat_template(prompt, add_generation_prompt=True)

        if self.model_type in ["florence2", "qwen2_vl"]:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        else:
            inputs = self.processor(questions=prompt, images=image, return_tensors="pt")

        inputs = self.data_to_dtype(send_to_device({**inputs}, self.device))
        
        with torch.inference_mode():
            generated_ids = self.unwrapped_model.generate(
                **inputs,
                max_length=self.max_length,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
                eos_token_id=self.processor.tokenizer.eos_token_id, 
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

            if self.model_type == "qwen2_vl":
                generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]

            generated_text = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            if self.model_type in ["qwen2_vl", "llava_qwen", "internvl2"]:
                generated_text = generated_text.replace("<|im_end|>", "")

        response_format = getattr(self.model.config, "response_format", "florence2")
        if response_format == "florence2":
            parsed_answer = self.florence2_processor.post_process_generation(
                generated_text, 
                task=self.task_prompt, 
                image_size=(image.width, image.height)
            )  # e.g.: {'<ROD>': {'polygons': [[[138.75201416015625, 796.1600341796875, 130.5600128173828, 799.2320556640625, 138.75201416015625, 851.4560546875, 146.94400024414062, 850.4320678710938]]], 'labels': ['bridge']}}
        elif response_format == "allseeing":
            parsed_answer = self.florence2_processor.post_process_generation(
                "".join([c + "<sep>".join(["".join([f"<loc_{v}>" for v in box]) for box in qboxes]) for c, qboxes in extract_objects(generated_text)[0].items()]), 
                task=self.task_prompt, 
                image_size=(image.width, image.height)
            )
        return parsed_answer

    @property
    def unwrapped_model(self):
        return self.model.module if self.wrap_with_ddp else self.model
    
    @property
    def model_config(self):
        return self.unwrapped_model.config

    def print_model_info(self):
        pass


class Florence2Inferencer(BaseInferencer):

    model_type = "florence2"
    CONFIG_CLASS = Florence2Config
    MODEL_CLASS = Florence2ForConditionalGeneration
    PROCESSOR_CLASS = Florence2Processor
        
    def print_model_info(self):
        _, all_param = get_num_parameters(self.unwrapped_model)
        _, vis_param = get_num_parameters(self.unwrapped_model.get_vision_tower())
        _, proj_param = get_num_parameters(self.unwrapped_model.get_mm_projection())
        _, lang_param = get_num_parameters(self.unwrapped_model.get_language_model())
        print(f"Model loaded from {self.model_ckpt_path}, "
              f"with {all_param:,d} parameters, including {vis_param:,d} "
              f"for vision tower, {proj_param:,d} for multimodal projection, "
              f"{lang_param:,d} for language model.")


class InternVL2Inferencer(BaseInferencer):

    model_type = "internvl2"
    CONFIG_CLASS = InternVLChatConfig
    MODEL_CLASS = InternVLChatModel
    PROCESSOR_CLASS = SimpleInternVL2Processor

    def print_model_info(self):
        _, all_param = get_num_parameters(self.unwrapped_model)
        _, vis_param = get_num_parameters(self.unwrapped_model.vision_model)
        _, proj_param = get_num_parameters(self.unwrapped_model.mlp1)
        _, lang_param = get_num_parameters(self.unwrapped_model.language_model)
        print(f"Model loaded from {self.model_ckpt_path}, "
              f"with {all_param:,d} parameters, including {vis_param:,d} "
              f"for vision tower, {proj_param:,d} for multimodal projection, "
              f"{lang_param:,d} for language model.")


class Qwen2VLInferencer(BaseInferencer):

    model_type = "qwen2_vl"
    CONFIG_CLASS = Qwen2VLConfig
    MODEL_CLASS = Qwen2VLForConditionalGeneration
    PROCESSOR_CLASS = Qwen2VLProcessor

    def print_model_info(self):
        _, all_param = get_num_parameters(self.unwrapped_model)
        _, vis_and_merger_param = get_num_parameters(self.unwrapped_model.visual)
        _, lang_param = get_num_parameters(self.unwrapped_model.model)
        print(f"Model loaded from {self.model_ckpt_path}, "
              f"with {all_param:,d} parameters, including {vis_and_merger_param:,d} "
              f"for vision tower and multimodal merger, "
              f"{lang_param:,d} for language model.")


class LlavaQwenInferencer(BaseInferencer):

    model_type = "llava_qwen"
    CONFIG_CLASS = LlavaQwenConfig
    MODEL_CLASS = LlavaQwenForCausalLM
    PROCESSOR_CLASS = SimpleLlavaQwenProcessor

    def print_model_info(self):
        _, all_param = get_num_parameters(self.unwrapped_model)
        _, vis_param = get_num_parameters(self.unwrapped_model.get_vision_tower())
        _, proj_param = get_num_parameters(self.unwrapped_model.get_model().mm_projector)
        _, resampler_param = get_num_parameters(self.unwrapped_model.get_model().vision_resampler)
        proj_param += resampler_param
        print(f"Model loaded from {self.model_ckpt_path}, "
              f"with {all_param:,d} parameters, including {vis_param:,d} "
              f"for vision tower, {proj_param:,d} for multimodal projection, ")
        

def get_inferencer(model_name_of_path, *args, **kwargs):
    if os.path.exists(model_name_of_path):
        model_type = json.load(open(os.path.join(model_name_of_path, "config.json"), "r"))["model_type"]
    else:
        model_type = AutoConfig.from_pretrained(model_name_of_path).model_type

    if model_type in ["internvl2", "internvl_chat"]:
        attn_implementation = kwargs.get("attn_implementation", None)
        if attn_implementation == "sdpa":
            print("InternVL2 does not support sdpa attention, using eager instead.")
            kwargs["attn_implementation"] = "eager"
        return InternVL2Inferencer(model_name_of_path, *args, **kwargs)
    elif model_type == "florence2":
        return Florence2Inferencer(model_name_of_path, *args, **kwargs)
    elif model_type == "qwen2_vl":
        return Qwen2VLInferencer(model_name_of_path, *args, **kwargs)
    elif model_type == "llava_qwen":
        return LlavaQwenInferencer(model_name_of_path, *args, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
