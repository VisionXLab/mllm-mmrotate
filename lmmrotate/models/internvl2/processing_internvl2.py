from typing import Union, Optional
from collections.abc import Sequence
from PIL import Image

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

from lmmrotate.models.internvl2.conversation import get_conv_template


logger = logging.get_logger(__name__)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN='<img>' 
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def process_image(pil_image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(pil_image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class SimpleInternVL2Processor(ProcessorMixin):
    """Unofficial! Only for lmmrotates."""

    template = "Hermes-2"
    num_image_token = 256
    attributes = ["tokenizer"]
    valid_kwargs = ["chat_template"]
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(tokenizer)

    def apply_chat_template(self, num_patches_list: Sequence[int], questions: Sequence[str], answers: Optional[Sequence[str]] = None):
        if answers is None:
            answers = [None] * len(questions)
        assert len(questions) == len(answers) == len(num_patches_list), (len(questions), len(answers), len(num_patches_list))

        queries = []
        for question, answer, num_patches in zip(questions, answers, num_patches_list):
            # if pixel_values is not None and '<image>' not in question:
            if num_patches > 0 and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], answer)
            query = template.get_prompt()

            if num_patches > 0:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)
            queries.append(query)
        return queries
    
    @staticmethod
    def process_pil_images(pil_images, input_size, max_num):
        all_pixel_values = []
        num_patches_list = []
        for pil_image in pil_images:
            all_pixel_values.append(process_image(pil_image, input_size, max_num))
            num_patches_list.append(all_pixel_values[-1].size(0))
        pixel_values = torch.cat(all_pixel_values, dim=0).to(torch.float16)
        return pixel_values, num_patches_list
    
    @property
    def img_context_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    
    def __call__(
        self,
        questions: Union[TextInput, PreTokenizedInput, Sequence[TextInput], Sequence[PreTokenizedInput]],
        answers: Union[TextInput, PreTokenizedInput, Sequence[TextInput], Sequence[PreTokenizedInput]] = None,
        apply_template: bool = True,
        images: Union[Image.Image, Sequence[Image.Image]] = None,
        num_patches_list: Optional[Sequence[int]] = None,
        input_size: int = 448,
        max_num: int = 12,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        padding_side: Optional[str] = None,
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

        if images is not None:
            assert num_patches_list is None, "num_patches_list should be None when images is not None"
            assert len(images) == len(questions), (len(images), len(questions))
            pixel_values, num_patches_list = self.process_pil_images(images, input_size, max_num)

            image_flags = [torch.tensor([1] * num_patches, dtype=torch.long) for num_patches in num_patches_list]
            image_flags = torch.concat(image_flags)
        elif num_patches_list is None:
            # treat as no image
            num_patches_list = [0] * len(questions)

        if apply_template:
            text_questions = self.apply_chat_template(num_patches_list, questions)
            if answers is not None:
                text_questions_answers = self.apply_chat_template(num_patches_list, questions, answers)
                text_questions_answers = [txt.rstrip("\n") for txt in text_questions_answers]

                text_answers = []
                for text_question, text_question_answer in zip(text_questions, text_questions_answers):
                    text_answers.append(text_question_answer.replace(text_question, ""))
                    assert "".join([text_question, text_answers[-1]]) == text_question_answer
                answers = text_answers
            questions = text_questions

        if padding_side is not None:
            self.tokenizer.padding_side = padding_side

        tokenizer_kwargs = {
            "padding": padding,
            "truncation": truncation,
            "return_tensors": return_tensors,
            **kwargs
        }
        if max_length is not None:
            tokenizer_kwargs["max_length"] = max_length

        text_inputs = self.tokenizer(
            questions,
            **tokenizer_kwargs
        )

        if answers is not None:
            additional_text_inputs = self.tokenizer(
                answers,
                **tokenizer_kwargs
            )

            input_ids = list(text_inputs["input_ids"])
            attention_mask = list(text_inputs["attention_mask"])
            input_ids = [ids[mask.bool()] for ids, mask in zip(input_ids, attention_mask)]

            additional_input_ids = list(additional_text_inputs["input_ids"])
            additional_attention_mask = list(additional_text_inputs["attention_mask"])
            additional_input_ids = [ids[mask.bool()] for ids, mask in zip(additional_input_ids, additional_attention_mask)]

            labels = []
            for i in range(len(input_ids)):
                labels.append(torch.cat([torch.full_like(input_ids[i], -100), additional_input_ids[i]], dim=0))
                input_ids[i] = torch.cat([input_ids[i], additional_input_ids[i]], dim=0)
            
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=-100)

            text_inputs = {
                "input_ids": input_ids, 
                "labels": labels, 
                "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            }

        if images is None:
            return BatchFeature(data={**text_inputs})
        if answers is None:
            return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})
        else:
            return BatchFeature(data={**text_inputs, "pixel_values": pixel_values, "image_flags": image_flags})


if __name__ == "__main__":
    processor = SimpleInternVL2Processor.from_pretrained("OpenGVLab/InternVL2-1B")
    processor.save_pretrained("./tmp_simple_internvl2_processor")
    processor_2 = SimpleInternVL2Processor.from_pretrained("./tmp_simple_internvl2_processor")
    import ipdb; ipdb.set_trace()
