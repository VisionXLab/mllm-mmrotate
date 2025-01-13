import json
from fire import Fire

import torch

from mmengine.fileio import load
from mmengine.evaluator import Evaluator

from mmrotate.utils import register_all_modules
from mmdet.utils import register_all_modules as register_all_modules_mmdet


def monkey_patch_of_collections_typehint_for_mmrotate1x():
    import collections
    from collections.abc import Mapping, Sequence, Iterable
    collections.Mapping = Mapping
    collections.Sequence = Sequence
    collections.Iterable = Iterable

monkey_patch_of_collections_typehint_for_mmrotate1x()

register_all_modules_mmdet(init_default_scope=False)
register_all_modules(init_default_scope=False)

# G=20
# THRESHOLD_TO_CHECK = [i / G for i in range(int(G))]
THRESHOLD_TO_CHECK = (0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.7)


def prepare_evaluator(dataset_name):
    if dataset_name == "dota":
        from mmrotate.evaluation import DOTAMetric
        evaluator = Evaluator(DOTAMetric(metric="mAP"))
        evaluator.dataset_meta = {
            'classes':
            ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
             'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
             'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
             'harbor', 'swimming-pool', 'helicopter'),
        }
    elif dataset_name == "fair":
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
    elif dataset_name == "dior":
        from mmrotate.evaluation import DOTAMetric
        evaluator = Evaluator(DOTAMetric(metric="mAP"))
        evaluator.dataset_meta = {
            'classes':
            ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
             'chimney', 'expressway-service-area', 'expressway-toll-station',
             'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
             'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'),
        }
    elif dataset_name == "srsdd":
        from mmrotate.evaluation import RotatedCocoMetric
        evaluator = Evaluator(RotatedCocoMetric(metric='bbox', classwise=True))
        evaluator.dataset_meta = {
            'classes':
            ('Cell-Container', 'Container', 'Dredger', 'Fishing', 'LawEnforce', 'ore-oil'),
        }
    elif dataset_name == "rsar":
        from mmrotate.evaluation import DOTAMetric
        evaluator = Evaluator(DOTAMetric(metric="mAP"))
        evaluator.dataset_meta = {'classes': ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')}
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset_name}")
    return evaluator


def get_results_of_different_thresholds(dataset_name, pickle_result_path):
    results = {}
    for threshold in THRESHOLD_TO_CHECK:
        print("".join(["="*10, f" score threshold={threshold} ", "="*10]))

        evaluator = prepare_evaluator(dataset_name)
        results_test = load(pickle_result_path)   
        for res in results_test:
            keep = res["pred_instances"]["scores"] > threshold
            res["pred_instances"]["labels"] = res["pred_instances"]["labels"][keep]
            res["pred_instances"]["bboxes"] = res["pred_instances"]["bboxes"][keep]
            res["pred_instances"]["scores"] = torch.ones_like(res["pred_instances"]["scores"][keep])
                
        mAP = evaluator.offline_evaluate(data_samples=results_test, chunk_size=128)
        mAP = mAP.get('dota/mAP', mAP.get('fair1m/mAP', mAP.get('r_coco/bbox_mAP_50')))
        
        results[threshold] = mAP
                
    print(json.dumps(results, indent=4))
    print("best result: ", max(results))
    print("best threshold: ", THRESHOLD_TO_CHECK[results.index(max(results))])


if __name__ == "__main__":
    Fire(get_results_of_different_thresholds)
