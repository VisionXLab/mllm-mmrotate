import copy
from typing import Union, Optional
from collections.abc import Sequence
from PIL import Image
from argparse import Namespace

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import logging, TensorType

from lmmrotate.models.llava.conversation import conv_templates
from lmmrotate.models.llava.mm_utils import process_images as llava_process_images, tokenizer_image_token
from lmmrotate.models.llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from lmmrotate.models.llava.multimodal_encoder.siglip_encoder import SigLipImageProcessor


logger = logging.get_logger(__name__)


def process_images(images, image_processor):
    model_cfg = Namespace(
        image_aspect_ratio="anyres_max_9", 
        image_grid_pinpoints=[
            [384, 384], [384, 768], [384, 1152], [384, 1536], [384, 1920], 
            [384, 2304], [768, 384], [768, 768], [768, 1152], [768, 1536], 
            [768, 1920], [768, 2304], [1152, 384], [1152, 768], [1152, 1152], 
            [1152, 1536], [1152, 1920], [1152, 2304], [1536, 384], [1536, 768], 
            [1536, 1152], [1536, 1536], [1536, 1920], [1536, 2304], [1920, 384], 
            [1920, 768], [1920, 1152], [1920, 1536], [1920, 1920], [1920, 2304], 
            [2304, 384], [2304, 768], [2304, 1152], [2304, 1536], [2304, 1920], 
            [2304, 2304]
        ], 
        image_crop_resolution=None, 
        image_split_resolution=None,
    )
    return llava_process_images(images, image_processor, model_cfg)


def preprocess_qwen(question, answer, tokenizer, has_image: bool = False, system_message: str = "You are a helpful assistant."):
    question = question.replace(DEFAULT_IMAGE_TOKEN, "").strip()
    if has_image:
        question = DEFAULT_IMAGE_TOKEN + "\n" + question
    sources = [[
        {"from": "human", "value": question},
        {"from": "gpt", "value": answer},
    ]]


    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    im_start, im_end = tokenizer.additional_special_tokens_ids
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return input_ids[0], targets[0]


class SimpleLlavaQwenProcessor(ProcessorMixin):
    """Unofficial! Only for lmmrotates."""

    template = "qwen_1_5"
    num_image_token = 256
    attributes = ["tokenizer"]
    valid_kwargs = ["chat_template"]
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(tokenizer)
        self.custom_image_processor = SigLipImageProcessor()

    def apply_chat_template(self, questions: Sequence[str], answers: Optional[Sequence[str]] = None, has_image: bool = True):
        if answers is None:
            answers = [None] * len(questions)
        assert len(questions) == len(answers), (len(questions), len(answers))

        queries = []
        for question, answer in zip(questions, answers):
            # if pixel_values is not None and '<image>' not in question:
            question = question.replace(DEFAULT_IMAGE_TOKEN, "").strip()
            if has_image:
                question = DEFAULT_IMAGE_TOKEN + "\n" + question

            conv = copy.deepcopy(conv_templates[self.template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], answer)
            query = conv.get_prompt()

            queries.append(query)
        return queries
    
    def __call__(
        self,
        questions: Union[TextInput, PreTokenizedInput, Sequence[TextInput], Sequence[PreTokenizedInput]],
        answers: Union[TextInput, PreTokenizedInput, Sequence[TextInput], Sequence[PreTokenizedInput]] = None,
        images: Union[Image.Image, Sequence[Image.Image]] = None,
        max_length: Optional[int] = None,
        **kwargs
    ) -> BatchFeature:
    
        if not (isinstance(questions, list) or isinstance(questions, tuple)):
            questions = [questions]

        if answers is not None and not (isinstance(answers, list) or isinstance(answers, tuple)):
            answers = [answers]
        
        if images is not None and not (isinstance(images, list) or isinstance(images, tuple)):
            images = [images]

        if answers is not None:
            assert len(questions) == len(answers), (len(questions), len(answers))

        has_image = False
        if images is not None:
            image_sizes = [image.size for image in images]
            modalities = ["image"] * len(images)
            images = process_images(images, self.custom_image_processor)
            vision_inputs = {"images": images, "modalities": modalities, "image_sizes": image_sizes}
            has_image = True

        self.tokenizer.padding_side = "right"

        if answers is not None:
            input_ids, labels = zip(*[preprocess_qwen(question, answer, self.tokenizer, has_image) for question, answer in zip(questions, answers)])

            input_ids = [_input_ids[:max_length] for _input_ids in input_ids]
            labels = [_labels[:max_length] for _labels in labels]

            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=-100)
            
            text_inputs = {
                "input_ids": input_ids, 
                "labels": labels, 
                "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            }
        else:
            prompt_questions = self.apply_chat_template(questions, has_image=has_image)
            if has_image:
                input_ids = [
                    tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    for prompt_question in prompt_questions
                ]
                input_ids = [_input_ids[:max_length] for _input_ids in input_ids]
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                text_inputs = {
                    # "input_ids": input_ids,  # i do not know why
                    "inputs": input_ids,
                    "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
                }
                return {**text_inputs, **vision_inputs}
            else:
                text_inputs = self.tokenizer(
                    prompt_questions,
                    max_length=max_length,
                    return_tensors="pt",
                    **kwargs
                )
                text_inputs = {**text_inputs}
                text_inputs["inputs"] = text_inputs.pop("input_ids")  # i do not know why
                return text_inputs

        if images is None:
            return BatchFeature(data={**text_inputs})
        return BatchFeature(data={**text_inputs, **vision_inputs})


if __name__ == "__main__":
    processor = SimpleLlavaQwenProcessor.from_pretrained("lmms-lab/llava-onevision-qwen2-0.5b-si")
    processor.save_pretrained("./tmp_simple_llava_qwen_processor")
    processor_2 = SimpleLlavaQwenProcessor.from_pretrained("./tmp_simple_llava_qwen_processor")
    import ipdb; ipdb.set_trace()
