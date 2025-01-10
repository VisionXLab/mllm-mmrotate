import pathlib
from typing import Optional
from dataclasses import dataclass, field

import transformers

from .dataset import OrientedDetSFTDataset
from .trainer import CustomTrainer, DatasetResetCallback, initialize_model_and_processor


@dataclass
class ModelArguments:
    model_type: Optional[str] = field(default="florence2")
    model_name_or_path: Optional[str] = field(default="microsoft/florence-2-large")
    image_square_length: Optional[int] = field(default=None)
    language_model_max_length: Optional[int] = field(default=None)
    model_revision: Optional[str] = field(default=None)
    language_model_lora: Optional[int] = field(default=None)
    
    
@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data.", "nargs": "+"})
    image_folder: Optional[str] = field(default=None, metadata={"nargs": "+"})
    dataset_mode: str = field(default="single")  # single / concat / balanced concat
    response_format: Optional[str] = field(default="florence2")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    freeze_language: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the language model.'},
    )
    freeze_vision: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the vision backbone of the model.'},
    )
    freeze_multimodal_projection: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the multimodal projection.'},
    )
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})


def parse_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.language_model_max_length = model_args.language_model_max_length
    data_args.model_type = model_args.model_type
    model_args.response_format = data_args.response_format
    return model_args, data_args, training_args


def train():
    # Initialize arguments
    model_args, data_args, training_args = parse_args()

    # Initialize model and processor
    model, processor = initialize_model_and_processor(model_args, training_args)
    
    # Initialize dataset
    train_dataset = OrientedDetSFTDataset(processor=processor, data_args=data_args, model=model)
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model, 
        processor=processor,
        tokenizer=processor.tokenizer,
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=train_dataset, 
        data_collator=train_dataset.collate_fn, 
        callbacks=[DatasetResetCallback(train_dataset)], 
    )
    
    # Train
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
        
    # Save final model
    trainer.save_final_model()


if __name__ == "__main__":
    train()
