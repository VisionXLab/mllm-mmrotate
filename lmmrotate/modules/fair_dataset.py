# https://github.com/yuyi1005/whollywood/blob/main/mmrotate/datasets/fair.py
import glob
import os.path as osp
import xml.etree.ElementTree as ET
from typing import List

import mmcv
from mmengine.dataset import BaseDataset

from mmrotate.registry import DATASETS
from mmrotate.datasets import DOTADataset


@DATASETS.register_module()
class FAIRDataset(BaseDataset):
    """FAIR dataset for detection.

    Args:
        ann_subdir (str): Subdir where annotations are.
            Defaults to 'Annotations/Oriented Bounding Boxes/'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        ann_type (str): Choose obb or hbb as ground truth.
            Defaults to `obb`.
    """

    METAINFO = {
        'classes':
        ('Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'C919', 'A220',
         'A321', 'A330', 'A350', 'ARJ21', 'Passenger Ship', 'Motorboat',
         'Fishing Boat', 'Tugboat', 'Engineering Ship', 'Liquid Cargo Ship',
         'Dry Cargo Ship', 'Warship', 'Small Car', 'Bus', 'Cargo Truck',
         'Dump Truck', 'Van', 'Trailer', 'Tractor', 'Excavator',
         'Truck Tractor', 'Basketball Court', 'Tennis Court', 'Football Field',
         'Baseball Field', 'Intersection', 'Roundabout', 'Bridge'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (22, 226, 252), (200, 182, 255),
         (22, 82, 0), (0, 246, 252), (182, 202, 255), (0, 102, 0),
         (255, 77, 155), (0, 226, 152), (182, 182, 155), (0, 82, 100),
         (120, 166, 254), (22, 226, 152), (200, 182, 155), (22, 82, 100)]
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based
        data_list = []
        if self.ann_file == '':
            img_files = glob.glob(
                osp.join(self.data_prefix['img_path'], '*.png'))
            for img_path in img_files:
                data_info = {}
                data_info['img_path'] = img_path
                img_name = osp.split(img_path)[1]
                data_info['file_name'] = img_name
                img_id = img_name[:-4]
                data_info['img_id'] = img_id
                img = mmcv.imread(img_path, backend='cv2')
                data_info['height'], data_info['width'] = img.shape[:2]

                instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
                data_info['instances'] = [instance]
                data_list.append(data_info)

            return data_list
        else:
            xml_files = glob.glob(osp.join(self.ann_file, '*.xml'))  # [:4000]
            if len(xml_files) == 0:
                raise ValueError('There is no xml file in ' f'{self.ann_file}')
            for xml_file in xml_files:
                data_info = {}
                img_id = osp.split(xml_file)[1][:-4]
                data_info['img_id'] = img_id
                img_name = img_id + '.png'
                data_info['file_name'] = img_name
                data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                                 img_name)

                raw_ann_info = ET.parse(xml_file)
                root = raw_ann_info.getroot()
                size = root.find('size')
                data_info['width'] = int(size.find('width').text)
                data_info['height'] = int(size.find('height').text)
                instances = []
                for obj in root.find('objects').findall('object'):
                    cls_name = obj.find('possibleresult').find('name').text
                    if cls_name not in cls_map.keys():
                        continue
                    instance = {}
                    poly = []
                    for p in obj.find('points')[0:4]:
                        xy = p.text.split(',')
                        poly.append(float(xy[0]))
                        poly.append(float(xy[1]))
                    instance['bbox'] = poly
                    instance['bbox_label'] = cls_map[cls_name]
                    instance['ignore_flag'] = 0
                    instances.append(instance)
                data_info['instances'] = instances
                data_list.append(data_info)

            return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get FAIR category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            List[int]: All categories in the image of specified index.
        """
        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]


@DATASETS.register_module()
class FAIRDOTADataset(DOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """

    METAINFO = {
        # 'classes':
        # ('Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'C919', 'A220',
        #  'A321', 'A330', 'A350', 'ARJ21', 'Passenger-Ship', 'Motorboat',
        #  'Fishing-Boat', 'Tugboat', 'Engineering-Ship', 'Liquid-Cargo-Ship',
        #  'Dry-Cargo-Ship', 'Warship', 'Small-Car', 'Bus', 'Cargo-Truck',
        #  'Dump-Truck', 'Van', 'Trailer', 'Tractor', 'Excavator',
        #  'Truck-Tractor', 'Basketball-Court', 'Tennis-Court', 'Football-Field',
        #  'Baseball-Field', 'Intersection', 'Roundabout', 'Bridge'),
        'classes':
        ('Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'C919', 'A220',
         'A321', 'A330', 'A350', 'ARJ21', 'Passenger Ship', 'Motorboat',
         'Fishing Boat', 'Tugboat', 'Engineering Ship', 'Liquid Cargo Ship',
         'Dry Cargo Ship', 'Warship', 'Small Car', 'Bus', 'Cargo Truck',
         'Dump Truck', 'Van', 'Trailer', 'Tractor', 'Excavator',
         'Truck Tractor', 'Basketball Court', 'Tennis Court', 'Football Field',
         'Baseball Field', 'Intersection', 'Roundabout', 'Bridge'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (22, 226, 252), (200, 182, 255),
         (22, 82, 0), (0, 246, 252), (182, 202, 255), (0, 102, 0),
         (255, 77, 155), (0, 226, 152), (182, 182, 155), (0, 82, 100),
         (120, 166, 254), (22, 226, 152), (200, 182, 155), (22, 82, 100)]
    }

    def __init__(self,
                 diff_thr: int = 100,
                 img_suffix: str = 'png',
                 **kwargs) -> None:
        self.diff_thr = diff_thr
        self.img_suffix = img_suffix
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based
        data_list = []
        if self.ann_file == '':
            img_files = glob.glob(
                osp.join(self.data_prefix['img_path'], f'*.{self.img_suffix}'))
            for img_path in img_files:
                data_info = {}
                data_info['img_path'] = img_path
                img_name = osp.split(img_path)[1]
                data_info['file_name'] = img_name
                img_id = img_name[:-4]
                data_info['img_id'] = img_id

                instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
                data_info['instances'] = [instance]
                data_list.append(data_info)

            return data_list
        else:
            txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
            if len(txt_files) == 0:
                raise ValueError('There is no txt file in '
                                 f'{self.ann_file}')
            for txt_file in txt_files:
                data_info = {}
                img_id = osp.split(txt_file)[1][:-4]
                data_info['img_id'] = img_id
                img_name = img_id + f'.{self.img_suffix}'
                data_info['file_name'] = img_name
                data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                                 img_name)

                instances = []
                with open(txt_file) as f:
                    s = f.readlines()
                    for si in s:
                        instance = {}
                        bbox_info = si.split()
                        instance['bbox'] = [float(i) for i in bbox_info[:8]]
                        cls_name = " ".join(bbox_info[8:-1])
                        if cls_name.startswith('other'):
                            continue
                        instance['bbox_label'] = cls_map[cls_name]
                        difficulty = int(bbox_info[-1])
                        if difficulty > self.diff_thr:
                            instance['ignore_flag'] = 1
                        else:
                            instance['ignore_flag'] = 0
                        instances.append(instance)
                data_info['instances'] = instances
                data_list.append(data_info)

            return data_list
