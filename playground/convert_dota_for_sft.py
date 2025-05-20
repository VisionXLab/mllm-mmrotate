"""
TODO in future versions:
- resolve the bounding problem (because Florence2 requires clip_0_999)
"""

import os
import sys
import math
import json
import numpy as np
import torch
import random
import shutil
from tqdm import tqdm
from typing import Union
from PIL import Image, ImageDraw
from lmmrotate.models import Florence2Processor


playground_path = "playground"
# playground_path = "../playground "


def normalize_polygon(polygon: list[int], clockwise: bool = True) -> list[tuple[int, int]]:
    """
    :param polygon: 输入为长度为8的list[int], 表示四边形的四个顶点 (x1, y1, x2, y2, x3, y3, x4, y4)。
    :param clockwise: 控制顶点排序的方向。True为顺时针, False为逆时针。
    """
    points = [(polygon[i], polygon[i + 1]) for i in range(0, 8, 2)]
    center = (sum(x for x, y in points) / 4, sum(y for x, y in points) / 4)
    angle_from_center = lambda point: math.atan2(point[1] - center[1], point[0] - center[0])
    points.sort(key=angle_from_center, reverse=not clockwise)
    start_index = points.index(min(points, key=lambda p: (p[1], p[0])))
    sorted_points = points[start_index:] + points[:start_index]
    return sorted_points


def clip_0_999(coords):
    return [min(max(coord, 0), 999) for coord in coords]


def get_processor():
    return Florence2Processor.from_pretrained("microsoft/Florence-2-large", revision="f92844072980ab91bc708ae2fc8c1227318023a4")


def main_convert_data_to_florence2fmt(ann_path, save_path, version=3):
    processor = get_processor()
    tokenizer = processor.tokenizer
    coordinates_quantizer = processor.post_processor.coordinates_quantizer
    quantize = lambda quad_box, width, height: coordinates_quantizer.quantize(
        coordinates=torch.tensor(np.array(quad_box).reshape(-1, 2)), size=(width, height),
    ).reshape(-1).tolist()

    with open(ann_path, "r") as fp:
        data = json.load(fp)
    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    categoryid2rank = {}
    categoryid2name = {}
    cat_names = sorted([_["name"] for _ in categories], key=lambda s: s.replace("-", " "))
    for cat in categories:
        category_id = cat["id"]
        category_name = cat["name"]
        categoryid2rank[category_id] = cat_names.index(category_name)
        categoryid2name[category_id] = category_name

    imgid2annlist = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in imgid2annlist:
            imgid2annlist[img_id] = []
        imgid2annlist[img_id].append(ann)

    new_data = []
    for img in images:
        img_id = img["id"]
        if img_id not in imgid2annlist:
            if version < 3:
                continue
            else:
                img["response"] = 'There are none.'
                new_data.append(img)
        else:
            img_ann_list = imgid2annlist[img_id]
            
            img_ann_list = [
                {
                    "polygon": clip_0_999(quantize(normalize_polygon(ann["segmentation"][0], False), width=img["width"], height=img["height"])),
                    "cat_rank": categoryid2rank[ann["category_id"]],
                    "cat_name": categoryid2name[ann["category_id"]].replace("-", " ").lower(),
                }
                for ann in img_ann_list
            ]  # simplify the annotation
            
            img_ann_list.sort(key=lambda ann: (
                ann["cat_rank"], ann["polygon"][1], ann["polygon"][0]
            ))
            
            response = ""
            last_cat_name = None
            for ann in img_ann_list:
                polygon = ann["polygon"]
                cat_name = ann["cat_name"]
                
                if version >= 2:
                    if cat_name != last_cat_name:
                        response += cat_name
                    else:
                        response += "<sep>"
                elif version == 1:
                    if cat_name != last_cat_name:
                        response += cat_name
                else:
                    raise ValueError(f"Invalid version: {version}")
                
                response += "".join(f"<loc_{v}>" for v in polygon)
                last_cat_name = cat_name
                
            img["response"] = response
            new_data.append(img)

    with open(save_path, "w") as fp:
        json.dump(new_data, fp)
        
    all_response = [_["response"] for _ in new_data]
    all_response_length = [len(tokenizer(_, return_tensors="pt")["input_ids"][0]) for _ in all_response]
    print("max response token number: ", max(all_response_length))
    print("min response token number: ", min(all_response_length))
    print("mean response token number: ", sum(all_response_length) / len(all_response_length))
    print("Samples whose token number is larger than 1024: ", sum(_ > 1024 for _ in all_response_length) / len(all_response_length))
    
    
def main_convert_to_llava_fmt(src_path, tgt_path):
    with open(src_path, "r") as fp:
        data = json.load(fp)
        
    with open(f"{playground_path}/data/prompts/oriented_detection_8parameters/prompts.txt", "r") as fp:
        prompts = fp.read().splitlines()
        
    new_data = []
    for sample in data:
        file_name = sample["file_name"]
        response = sample["response"]
        
        new_sample = {
            "id": file_name, 
            "image": file_name,
            "conversations": [
                {
                    "from": "human",
                    "value": "".join(["<image>\n", random.choice(prompts)])
                },
                {
                    "from": "gpt",
                    "value": response
                }
            ],
            "data_source": "florence-dota"
        }
        new_data.append(new_sample)
        
    with open(tgt_path, "w") as fp:
        json.dump(new_data, fp, indent=2)
    print(f"Save to {tgt_path}")
    
    
def test_visualize_converted_data(ann_path, image_folder, version=3):
    processor = get_processor()
    with open(ann_path, "r") as fp:
        data = json.load(fp)
    data = random.sample(data, 20)
    
    def draw_quad(image_path: str, coords: Union[str, list[float]]):
        if isinstance(coords, str):
            coords = list(map(float, coords.split(',')))
        all_polygons = []
        for i in range(len(coords) // 8):
            all_polygons.append(coords[i * 8: (i + 1) * 8])
        print(f"left {coords[(i + 1) * 8:]}")
        
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for polygon in all_polygons:
            draw.polygon(polygon, outline="red", width=3)
        return image
    
    for sample in data:
        file_name = sample["file_name"]
        response = sample["response"]
        
        if response == 'There are none.':
            continue
        
        width = sample["width"]
        height = sample ["height"]
        
        parsed_answer = processor.post_process_generation(
            response, 
            task="<ROD>", 
            image_size=(width, height)
        )

        res_str = ", ".join(str(n) for c in parsed_answer["<ROD>"]["polygons"] for l in c for n in l)
        image = draw_quad(f"{image_folder}/{file_name}", res_str)
        os.makedirs("tmp_vis", exist_ok=True)
        image.save(f"tmp_vis/{file_name}")


def filter_train_split_from_florence_dota(florence_trainval_path, dota_train_image_folder, save_path):
    # python scripts_py/convert_dota_for_sft.py filter_train_split_from_florence_dota playground/data/florence-dota/florence_split_ss_dota_trainval_v2.json playground/data/DOTA/train/images/ playground/data/florence-dota/florence_split_ss_dota_train_v2.json
    with open(florence_trainval_path, "r") as fp:
        data = json.load(fp)

    prefix_list = [fname.rstrip('.png') for fname in os.listdir(dota_train_image_folder)]

    def check_filename_in_prefixlist(filename):
        return filename[:5] in prefix_list

    filtered_data = [d for d in data if check_filename_in_prefixlist(d["file_name"])]
    print(f"{len(data)} -> {len(filtered_data)} samples.")

    with open(save_path, "w") as fp:
        json.dump(filtered_data, fp)
    print(f"Save to {save_path}")


def filter_seperate_split_from_split_dota(split_dota_annfiles_path, dota_train_image_folder, dota_val_image_folder, save_path_train, save_path_val):
    # python scripts_py/convert_dota_for_sft.py filter_seperate_split_from_split_dota playground/data/split_ss_dota/trainval/annfiles playground/data/DOTA/train/images/ playground/data/DOTA/val/images/ playground/data/split_ss_dota/train/annfiles playground/data/split_ss_dota/val/annfiles
    trainval_annfiles = list(os.listdir(split_dota_annfiles_path))
    train_prefix_list = [fname.rstrip('.png') for fname in os.listdir(dota_train_image_folder)]
    val_prefix_list = [fname.rstrip('.png') for fname in os.listdir(dota_val_image_folder)]

    def check_filename_in_prefixlist(filename, prefix_list):
        return filename[:5] in prefix_list

    train_annfiles = []
    val_annfiles = []
    for f in trainval_annfiles:
        if check_filename_in_prefixlist(f, train_prefix_list):
            train_annfiles.append(f)
        elif check_filename_in_prefixlist(f, val_prefix_list):
            val_annfiles.append(f)
        else:
            raise ValueError(f"Invalid filename: {f}")
    
    print(f"{len(trainval_annfiles)} samples -> {len(train_annfiles)} train and {len(val_annfiles)} val samples.")

    os.makedirs(save_path_train, exist_ok=True)
    for annfile in tqdm(train_annfiles):
        shutil.copyfile(os.path.join(split_dota_annfiles_path, annfile),
                        os.path.join(save_path_train, annfile))
        
    os.makedirs(save_path_val, exist_ok=True)
    for annfile in tqdm(val_annfiles):
        shutil.copyfile(os.path.join(split_dota_annfiles_path, annfile),
                        os.path.join(save_path_val, annfile))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        func_name = sys.argv[1]
        func = globals().get(func_name)
        if func is not None:
            if len(sys.argv) > 2:
                func(*sys.argv[2:])
            else:
                func()
        else:
            print(f"Function {func_name} not found.")
    else:
        # python scripts_py/convert_dota_for_sft.py
        version = 3

        # DOTA
        ann_path = f"{playground_path}/data/split_ss_dota/trainval.json"
        save_path = f"{playground_path}/data/florence-dota/florence_split_ss_dota_trainval_v{version}.json"
        image_folder = f"{playground_path}/data/split_ss_dota/trainval/images"
        main_convert_data_to_florence2fmt(ann_path, save_path, version=version)
        test_visualize_converted_data(save_path, image_folder, version=version)

        # FAIR1M1.0
        ann_path = f"{playground_path}/data/split_ss_fair1m_1_0/train/train.json"
        save_path = f"{playground_path}/data/florence-dota/florence_split_ss_fair1m_1_0_train_v{version}.json"
        image_folder = f"{playground_path}/data/split_ss_fair1m_1_0/train/images"
        main_convert_data_to_florence2fmt(ann_path, save_path, version=version)
        test_visualize_converted_data(save_path, image_folder, version=version)

        # DIOR
        ann_path = f"{playground_path}/data/DIOR/Annotations/trainval.json"
        save_path = f"{playground_path}/data/florence-dota/florence_dior_r_trainval_v{version}.json"
        image_folder = f"{playground_path}/data/DIOR/JPEGImages-trainval"
        main_convert_data_to_florence2fmt(ann_path, save_path, version=version)
        test_visualize_converted_data(save_path, image_folder, version=version)

        # SRSDD
        ann_path = f"{playground_path}/data/SRSDD/train.json"
        save_path = f"{playground_path}/data/florence-dota/florence_srsdd_train_v{version}.json"
        image_folder = f"{playground_path}/data/SRSDD/train/images"
        main_convert_data_to_florence2fmt(ann_path, save_path, version=version)
        test_visualize_converted_data(save_path, image_folder, version=version)
