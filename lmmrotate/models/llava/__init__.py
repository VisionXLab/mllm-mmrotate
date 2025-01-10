# Copied from https://github.com/LLaVA-VL/LLaVA-NeXT/tree/79ef45a6d8b89b92d7a8525f077c3a3a9894a87d/llava/model  # noqa
from .builder import load_pretrained_model
from .language_model import (
    LlavaLlamaForCausalLM, LlavaQwenForCausalLM, LlavaMistralForCausalLM, LlavaMixtralForCausalLM, 
    LlavaQwenMoeForCausalLM, LlavaGemmaForCausalLM, LlavaMptForCausalLM,
    LlavaLlamaConfig, LlavaQwenConfig, LlavaMistralConfig, LlavaMixtralConfig,
    LlavaQwenMoeConfig, LlavaGemmaConfig, LlavaMptConfig,
)
from .processing_llava import SimpleLlavaQwenProcessor

__all__ = [
    "load_pretrained_model",
    "LlavaLlamaForCausalLM",
    "LlavaQwenForCausalLM",
    "LlavaMistralForCausalLM",
    "LlavaMixtralForCausalLM",
    "LlavaQwenMoeForCausalLM",
    "LlavaGemmaForCausalLM",
    "LlavaMptForCausalLM",
    "LlavaLlamaConfig",
    "LlavaQwenConfig",
    "LlavaMistralConfig",
    "LlavaMixtralConfig",
    "LlavaQwenMoeConfig",
    "LlavaGemmaConfig",
    "LlavaMptConfig",
    "SimpleLlavaQwenProcessor",
]

AVAILABLE_MODELS = {
    # "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    # "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
    # "llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
    # "llava_qwen_moe": "LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig",    
    # Add other models as needed
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except Exception as e:
        print(f"Failed to import {model_name} from llava.language_model.{model_name}. Error: {e}")
