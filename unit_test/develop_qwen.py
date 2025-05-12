from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from lmmrotate.dataset import OrientedDetSFTDataset
from argparse import Namespace
from torch.utils.data import DataLoader


args = Namespace(
    model_type="qwen2_vl",
    model_name_or_path="Qwen/Qwen2-VL-2B-Instruct",
    data_path=["./playground/data/florence-dota/florence_dior_r_trainval_v2.json"],
    image_folder=["./playground/data/DIOR/JPEGImages-trainval"],
    language_model_max_length=2048,
    model_revision=None,
    dataset_mode='single',
    response_format='florence2'
)

model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_name_or_path)
processor = Qwen2VLProcessor.from_pretrained(args.model_name_or_path)

NUM_BBOX_BINS = 1000
coordinate_tokens = [f"<loc_{i}>" for i in range(NUM_BBOX_BINS)] + ["<sep>"]
num_new_tokens = processor.tokenizer.add_tokens(coordinate_tokens, special_tokens=True)
model.resize_token_embeddings(len(processor.tokenizer))

train_dataset = OrientedDetSFTDataset(processor=processor, data_args=args)

dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=train_dataset.collate_fn, num_workers=0)
dataiter = iter(dataloader)

losses = []
for _ in range(10):
    batch = next(dataiter)

    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    pixel_values = batch['pixel_values']
    image_grid_thw = batch['image_grid_thw']
    labels = batch['labels']

    print("input_ids_decoded")
    input_ids_decoded = train_dataset.processor.tokenizer.batch_decode(input_ids)
    print(input_ids_decoded)

    print("labels_decoded")
    converted_labels = []
    for label in labels:
        converted_label = label.clone()
        converted_label[converted_label == -100] = 0
        converted_labels.append(converted_label)
    converted_labels = [label.tolist() for label in converted_labels]
    labels_decoded = train_dataset.processor.tokenizer.batch_decode(converted_labels)
    print(labels_decoded)

    print("pure_labels_decoded")
    pure_labels = [label[label != -100].tolist() for label in labels]
    pure_labels_decoded = train_dataset.processor.tokenizer.batch_decode(pure_labels)
    print(pure_labels_decoded)

    loss = model(**batch).loss
    losses.append(loss)

    import ipdb; ipdb.set_trace()