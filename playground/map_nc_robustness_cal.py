# Use rotated-fcos-le90_r50_fpn_1x_dior-2 to draw the plot
import os
import pandas
import random
from fire import Fire

import numpy as np

import torch
from mmengine.fileio import load
from mmengine.evaluator import Evaluator


def prepare_evaluator(dataset_name):
    if "dota" in dataset_name:
        from mmrotate.evaluation import DOTAMetric
        evaluator = Evaluator(DOTAMetric(metric="mAP"))
        evaluator.dataset_meta = {
            'classes':
            ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
             'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
             'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
             'harbor', 'swimming-pool', 'helicopter'),
        }
    elif "fair" in dataset_name:
        from lmmrotate.modules.fair_metric import FAIRMetric
        evaluator = Evaluator(FAIRMetric(metric="mAP"))
        evaluator.dataset_meta = {
            'classes':
            ('Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'C919', 'A220',
            'A321', 'A330', 'A350', 'ARJ21', 'Passenger Ship', 'Motorboat',
            'Fishing Boat', 'Tugboat', 'Engineering Ship', 'Liquid Cargo Ship',
            'Dry Cargo Ship', 'Warship', 'Small Car', 'Bus', 'Cargo Truck',
            'Dump Truck', 'Van', 'Trailer', 'Tractor', 'Excavator',
            'Truck Tractor', 'Basketball Court', 'Tennis Court', 'Football Field',
            'Baseball Field', 'Intersection', 'Roundabout', 'Bridge'),
        }
    elif "dior" in dataset_name:
        from mmrotate.evaluation import DOTAMetric
        evaluator = Evaluator(DOTAMetric(metric="mAP"))
        evaluator.dataset_meta = {
            'classes':
            ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
             'chimney', 'expressway-service-area', 'expressway-toll-station',
             'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
             'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'),
        }
    elif "srsdd" in dataset_name:
        from mmrotate.evaluation import RotatedCocoMetric
        evaluator = Evaluator(RotatedCocoMetric(metric='bbox', classwise=True))
        evaluator.dataset_meta = {
            'classes':
            ('Cell-Container', 'Container', 'Dredger', 'Fishing', 'LawEnforce', 'ore-oil'),
        }
    return evaluator


def main(dataset_name: str, pickle_result_path: str, csv_result_path: str, g: int=20):
    results = {}

    thresholds = [i / g for i in range(int(g))]
    for threshold in thresholds:
        for hack_score in ["score", "1", 
                           "rnd:42", "rnd:666", "rnd:6666", "rnd:66666", "rnd:666666", "rnd:2024", 
                           "rnd:9968", "rnd:0", "rnd:1", "rnd:123"]:
            print("".join(["="*10, f" score threshold={threshold} ", "="*10]))

            evaluator = prepare_evaluator(dataset_name)

            results_test = load(pickle_result_path)   
            for res in results_test:
                keep = res["pred_instances"]["scores"] > threshold
                res["pred_instances"]["scores"] = res["pred_instances"]["scores"][keep]
                res["pred_instances"]["labels"] = res["pred_instances"]["labels"][keep]
                res["pred_instances"]["bboxes"] = res["pred_instances"]["bboxes"][keep]
                
                if hack_score == "1":
                    res["pred_instances"]["scores"] = torch.ones_like(res["pred_instances"]["scores"])
                elif hack_score.startswith("rnd:"):
                    seed = hack_score.split(":")[-1]
                    random.seed(seed)
                    torch.manual_seed(seed)
                    res["pred_instances"]["scores"] = torch.rand_like(res["pred_instances"]["scores"])
                    
            map = evaluator.offline_evaluate(data_samples=results_test, chunk_size=128)['dota/mAP']
            
            if threshold not in results:
                results[threshold] = {}
            results[threshold][hack_score] = map
                
        random_values = [v for k, v in results[threshold].items() if k.startswith("rnd:")]
        results[threshold]["rnd:max"] = max(random_values)
        results[threshold]["rnd:min"] = min(random_values)
        results[threshold]["rnd:mean"] = sum(random_values) / len(random_values)
                
    for threshold in thresholds:
        random_values = [v for k, v in results[threshold].items() if k.startswith("rnd:")]
        results[threshold]["rnd:mean"] = np.mean(random_values)
        results[threshold]["rnd:std"] = np.std(random_values)

    df = pandas.DataFrame.from_dict(results).T
    df.to_csv(csv_result_path)
    print(df)


if __name__ == "__main__":
    # python -u scripts_py/map_nc_robustness_cal.py --dataset_name dota_train --pickle_result_path playground/mmrotate_workdir/rotated-retinanet-rbox-le90_r50_fpn_1x_dota-train/results_val.pkl --csv_result_path playground/mmrotate_workdir/rotated-retinanet-rbox-le90_r50_fpn_1x_dota-train/results_val_score_threshold.csv
    # python -u scripts_py/map_nc_robustness_cal.py --dataset_name dota_train --pickle_result_path playground/mmrotate_workdir/rotated-fcos-le90_r50_fpn_1x_dota-train/results_val.pkl --csv_result_path playground/mmrotate_workdir/rotated-fcos-le90_r50_fpn_1x_dota-train/results_val_score_threshold.csv
    # python -u scripts_py/map_nc_robustness_cal.py --dataset_name dior --pickle_result_path scripts_py/eval_mmrotate/rotated-retinanet-rbox-le90_r50_fpn_1x_dior/rotated-retinanet-rbox-le90_r50_fpn_1x_dior/results_test.pkl --csv_result_path scripts_py/eval_mmrotate/rotated-retinanet-rbox-le90_r50_fpn_1x_dior/rotated-retinanet-rbox-le90_r50_fpn_1x_dior/results_test_score_threshold.csv
    # python -u scripts_py/map_nc_robustness_cal.py --dataset_name dior --pickle_result_path playground/mmrotate_workdir/rotated-fcos-le90_r50_fpn_1x_dior-2/results_test.pkl --csv_result_path playground/mmrotate_workdir/rotated-fcos-le90_r50_fpn_1x_dior-2/results_test_score_threshold.csv
    Fire(main)
