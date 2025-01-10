import os
import re
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
from rapidfuzz import process  # pip install rapidfuzz

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import mmcv
from mmengine import dump
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData
from mmengine.dataset import default_collate
from mmdet.models.utils import samplelist_boxtype2tensor
from mmrotate.structures import QuadriBoxes
from mmrotate.visualization import RotLocalVisualizer

from lmmrotate.dataset import OrientedDetEvalDataset
from lmmrotate.models import get_inferencer
from lmmrotate.utils import (init_distributed_device, world_info_from_env, 
                             monkey_patch_of_collections_typehint_for_mmrotate1x)


monkey_patch_of_collections_typehint_for_mmrotate1x()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
dataloader_num_workers = int(os.environ.get("NUM_WORKERS", 8))


def parse_args():
    parser = argparse.ArgumentParser()
    # model options
    parser.add_argument("--model_ckpt_path", type=str, required=True, nargs="+")
    parser.add_argument("--torch_dtype", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    # eval options
    parser.add_argument("--result_path", type=str, default=None, nargs="+")
    parser.add_argument("--eval_intermediate_checkpoints", action="store_true", default=False)
    parser.add_argument("--vis", action="store_true", default=False)
    parser.add_argument("--pass_evaluate", action="store_true", default=False)
    # dataset options
    parser.add_argument("--dataset_type", type=str, default=None, choices=["dota", "dior", "fair1m", "srsdd", "dota_train", "fair1m_2.0_train"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split", type=str, default=["trainval", "test"], choices=["trainval", "test"], nargs="+")
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--clip_num", type=int, default=None)
    # folder name
    parser.add_argument("--folder_name", type=str, default="eval")
    args = parser.parse_args()

    if args.dataset_type is None:
        def determine_dataset_type(path):
            if "dota" in path and "train" in path and "trainval" not in path:
                return "dota_train"  # train on `train` split, eval on `val` split
            elif "dota" in path:
                return "dota"  # train on `trainval` split, eval on `test` split
            elif "dior" in path:
                return "dior"  # train on `trainval` split, eval on `test` split
            elif "fair1m" in path and "train" in path and "2.0" in path and "trainval" not in path:
                return "fair1m_2.0_train"  # train on `train` split of 1.0/2.0 (consistent), eval on `val` split of 2.0 (newly added)
            elif "fair1m" in path and "2.0" in path:  # TODO: NotImplement Yet
                return "fair1m_2.0"  # train on `trainval` split of 2.0 (consistent train and additional val comparing to 1.0), eval on `test` split of 2.0 (extended)
            elif "fair1m" in path:
                return "fair1m"  # train on `trainval` split of 1.0, eval on `test` split of 1.0
            elif "srsdd" in path:
                return "srsdd"  # train on `train` split, eval on `test` split
        dataset_type_list = [determine_dataset_type(p) for p in args.model_ckpt_path]
        assert all([d == dataset_type_list[0] for d in dataset_type_list]), \
            f"Dataset type should be the same for all model_ckpt_path {args.model_ckpt_path}"
        args.dataset_type = dataset_type_list[0]
    
    if args.eval_intermediate_checkpoints:
        new_model_ckpt_path = []
        new_result_path = []
        for path in args.model_ckpt_path:
            new_model_ckpt_path.append(path)
            new_result_path.append(os.path.join(path, args.folder_name))
            checkpoint_folder_list = [
                f for f in os.listdir(path) 
                if re.match(r"checkpoint-\d+", f)
            ]
            new_model_ckpt_path.extend([
                os.path.join(path, f) 
                for f in checkpoint_folder_list
            ])
            new_result_path.extend([
                os.path.join(path, f, args.folder_name) 
                for f in checkpoint_folder_list
            ])
        
        args.model_ckpt_path = new_model_ckpt_path
        args.result_path = new_result_path
        print(f"Found {len(new_model_ckpt_path)} intermediate checkpoints")

    if args.result_path is not None:
        assert len(args.result_path) == len(args.model_ckpt_path), \
            f"Length of result_path {args.result_path} should be the same as model_ckpt_path {args.model_ckpt_path}"
    else:
        args.result_path = [os.path.join(p, f'{args.folder_name}_{args.dataset_type}') for p in args.model_ckpt_path]
    return args
    

def postprocess_parsed_answer(parsed_answer, cls_map, logger=None):
    try:
        parsed_answer = parsed_answer['<ROD>']
        polygons = parsed_answer["polygons"]
        labels = parsed_answer["labels"]
        
        all_polygons = []
        all_scores = []
        all_labels = []
        for one_cat_polygons, one_cat_label in zip(polygons, labels):
            if one_cat_label.strip() == "":
                continue
            
            if one_cat_label in cls_map:
                label = cls_map[one_cat_label.lower()]
            else:
                fuzzy_matched_cat = process.extractOne(one_cat_label.lower(), cls_map.keys())[0]
                if logger is None:
                    print(f"Fuzzy matched {one_cat_label.lower()} to {fuzzy_matched_cat}")
                else:
                    logger.info(f"Fuzzy matched {one_cat_label.lower()} to {fuzzy_matched_cat}")
                label = cls_map[fuzzy_matched_cat]
            
            for coords in one_cat_polygons:
                if len(coords) == 8:
                    all_polygons.append(coords)
                    all_labels.append(label)
                    all_scores.append(1.0)
                else:
                    for i in range(len(coords) // 8):
                        all_polygons.append(coords[i * 8: (i + 1) * 8])
                        all_labels.append(label)
                        all_scores.append(1.0)
        
        results = InstanceData()
        results.bboxes = QuadriBoxes(all_polygons).convert_to("rbox")
        results.labels = torch.as_tensor(all_labels)
        results.scores = torch.as_tensor(all_scores)
        return results
    except Exception as err:
        print(err)
        import ipdb; ipdb.set_trace()
    
    
def inference(model_ckpt_path, answers_path, dataset=None, device=None, torch_dtype=None, attn_implementation="flash_attention_2"):
    _, rank, world_size, _ = world_info_from_env()
    
    print(f"Rank {rank} of {world_size}")
    if device is None:
        device = torch.device(f"cuda:{rank}")
    
    if dataset is None:
        dataset = OrientedDetEvalDataset()
    
    model = get_inferencer(model_ckpt_path, device=device, torch_dtype=torch_dtype, attn_implementation=attn_implementation)
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    else:
        sampler = None
    dataloader = DataLoader(dataset, num_workers=dataloader_num_workers, sampler=sampler, collate_fn=default_collate)
    
    all_parsed_answers = {}
    for img_path, data_sample in tqdm(dataloader, disable=(rank!=0)):
        img_path, data_sample = img_path[0], data_sample[0]  # when using dataloader and batch_size=1
        all_parsed_answers[data_sample.file_name] = model(img_path)
        
    all_ranks_parsed_answers = [None for _ in range(world_size)]
    print(f"Rank {rank} is done")
    dist.barrier()
    print(f"Rank {rank} is gathering")
    dist.all_gather_object(all_ranks_parsed_answers, all_parsed_answers)

    if rank == 0:
        all_parsed_answers = {
            k: v 
            for each_rank_parsed_answers in all_ranks_parsed_answers 
            for k, v in each_rank_parsed_answers.items()
        }
        
        with open(answers_path, 'w') as f:
            json.dump(all_parsed_answers, f, indent=4, ensure_ascii=False)


def get_evaluator(dataset_type, is_test_set, results_path=None):
    def set_split_attr(evaluator, split_attr):
        evaluator.split = split_attr
        return evaluator

    def _get_submission_evaluator(METRIC_CLASS, _results_path):
        return set_split_attr(METRIC_CLASS(
            format_only=True,
            merge_patches=True,
            outfile_prefix=f'{_results_path}/dota_Task1'
        ), "test")

    if dataset_type == "fair1m":
        from lmmrotate.modules.fair_metric import FAIRMetric
        if is_test_set:
            return _get_submission_evaluator(FAIRMetric, results_path)
        else:
            return set_split_attr(FAIRMetric(metric="mAP"), "train")
    elif dataset_type == "fair1m_2.0_train":
        from lmmrotate.modules.fair_metric import FAIRMetric
        return set_split_attr(FAIRMetric(metric="mAP"), "val" if is_test_set else "train")
    elif dataset_type == "dota":
        from mmrotate.evaluation import DOTAMetric
        if is_test_set:
            return _get_submission_evaluator(DOTAMetric, results_path)
        else:
            return set_split_attr(DOTAMetric(metric="mAP"), "trainval")
    elif dataset_type == "dota_train":
        from mmrotate.evaluation import DOTAMetric
        return set_split_attr(DOTAMetric(metric="mAP"), "val" if is_test_set else "train")
    elif dataset_type == "dior":
        from mmrotate.evaluation import DOTAMetric
        return set_split_attr(DOTAMetric(metric="mAP"), "test" if is_test_set else "trainval")
    elif dataset_type == "srsdd":
        from mmrotate.evaluation import DOTAMetric
        return set_split_attr(DOTAMetric(metric="mAP"), "test" if is_test_set else "train")
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")
          
            
def evaluate_results(answers_path, dataset=None, vis_root=None):
    if dataset is None:
        dataset = OrientedDetEvalDataset()
        
    if vis_root is not None:
        os.makedirs(vis_root, exist_ok=True)
        visualizer = RotLocalVisualizer(name='visualizer', vis_backends=[dict(type='LocalVisBackend')])
        visualizer.dataset_meta = dataset.metainfo
    
    with open(answers_path, "r") as f:
        all_parsed_answers = json.load(f)
        
    results_path = answers_path.split(".")[0]
    evaluator = get_evaluator(dataset.dataset_type, dataset.is_test_set, results_path)
    evaluator.dataset_meta = dataset.metainfo
    
    os.makedirs(results_path, exist_ok=True)
    logger = MMLogger.get_instance(
        answers_path, logger_name=f'{dataset.dataset_type} {evaluator.split} Evaluation', distributed=True,
        log_file=f'{results_path}/eval.log', log_level='INFO', file_mode="a")
    
    logger.info("-" * 20)
    pickle_results = []
    for img_path, data_sample in dataset:
        parsed_answer = all_parsed_answers[data_sample.file_name]
        pred_instances = postprocess_parsed_answer(parsed_answer, dataset.cls_map, logger)
        data_sample.pred_instances = pred_instances
        samplelist_boxtype2tensor([data_sample])
        
        data_sample_dict = data_sample.to_dict()
        pickle_results.append(data_sample_dict)
        evaluator.process(data_batch=None, data_samples=[data_sample_dict])
        
        if vis_root is not None:
            img = mmcv.imread(img_path)
            img = mmcv.imconvert(img, 'bgr', 'rgb')
            visualizer.add_datasample(
                'result',
                img,
                data_sample=data_sample,
                out_file=os.path.join(vis_root, data_sample.file_name),
                pred_score_thr=0)
            
    evaluator.compute_metrics(evaluator.results)
    logger.info("-" * 20 + '\n')  

    pickle_results_path = f"{results_path}/output.pkl"
    dump(pickle_results, pickle_results_path)
    logger.info(f"Dumped results to {pickle_results_path}")  


if __name__ == "__main__":
    args = parse_args()

    device = init_distributed_device(dist_backend=None)  # using nccl may raise error for 4090 cluster
    _, rank, _, _ = world_info_from_env()
    
    datasets = {}
    for split in args.split:
        if split == "test":
            dataset = OrientedDetEvalDataset(dataset_type=args.dataset_type,
                                             data_root=args.data_root, 
                                             shuffle_seed=args.shuffle_seed, 
                                             clip_num=args.clip_num, 
                                             is_test_set=True)
        elif split == "trainval":
            dataset = OrientedDetEvalDataset(dataset_type=args.dataset_type,
                                             data_root=args.data_root, 
                                             shuffle_seed=args.shuffle_seed, 
                                             clip_num=args.clip_num, 
                                             is_test_set=False)
        else:
            raise ValueError(f"Unknown split {split}")
        datasets[split] = dataset
    
    evaluate_todos = defaultdict(list)
    for idx, (model_ckpt_path, result_path) in enumerate(zip(args.model_ckpt_path, args.result_path)):
        print(f"[{idx} / {len(args.model_ckpt_path)}] Inference for {model_ckpt_path}")
        for split, dataset in datasets.items():
            os.makedirs(result_path, exist_ok=True)
            
            split_surfix = split
            if args.clip_num is not None:
                split_surfix = f"{split}_seed{args.shuffle_seed}_{args.clip_num}samples"
            answers_path = f"{result_path}/parsed_answers_{split_surfix}.json"
            vis_root = f"{result_path}/parsed_answers_{split_surfix}/" if args.vis else None
            
            if not os.path.exists(answers_path):
                inference(model_ckpt_path, answers_path, dataset, device, args.torch_dtype, args.attn_implementation)
            else:
                print(f"Skip finished inference for {model_ckpt_path} on {split}")
                
            evaluate_todos[split].append((answers_path, vis_root))

    if rank == 0 and not args.pass_evaluate:
        for split, todos in evaluate_todos.items():
            for answers_path, vis_root in todos:
                evaluate_results(answers_path, datasets[split], vis_root)
                
    dist.barrier()
    dist.destroy_process_group()
