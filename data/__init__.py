from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve, auc
from detnet.data import COCODetection
from detnet.transforms.train_transforms import ToRGB, expand_dataset_by_flips
from .froc import load_ground_truth, Prediction, compute_froc


class Dataset(COCODetection):
    def __init__(self, images_root, annotations_json, annotations_csv):
        super(Dataset, self).__init__(images_root, annotations_json)
        self.classification_gt = [len(self.coco.imgToAnns[image_id]) > 0 for image_id in self.ids]

        self.localization_gt = load_ground_truth(annotations_csv)

    def prediction_to_classification_and_localization(self, predictions):
        classification = {}
        localization = {}
        for image_id in self.ids:
            pred = predictions[str(image_id)]
            bbox = pred[0]
            if len(bbox) == 0:
                classification[image_id] = 0
                localization[image_id] = bbox
            else:
                classification[image_id] = max(bbox[:, 0])
                img = self.coco.imgs[image_id]
                width, height = img['width'], img['height']
                scale = np.asarray([1, width, height, width, height])
                localization[image_id] = bbox * scale

        return classification, localization

    def _update_coco_by_ids(self):
        super(Dataset, self)._update_coco_by_ids()
        self.classification_gt = [len(self.coco.imgToAnns[image_id]) > 0 for image_id in self.ids]

        num_image, num_object, object_dict = self.localization_gt
        object_dict = {k: v for k, v in object_dict.items() if k in self.ids}
        num_image = len(object_dict)
        num_object = sum([len(v) for v in object_dict.values()])
        self.localization_gt = num_image, num_object, object_dict

    def evaluate(self, predictions, num_processes=1):
        coco_metric = COCODetection.evaluate(self, predictions, num_processes=num_processes)

        classification, localization = self.prediction_to_classification_and_localization(predictions)
        classification = [classification[i] for i in self.ids]
        fpr, tpr, _ = roc_curve(self.classification_gt, classification)
        roc_auc = auc(fpr, tpr)
        print('AUC:', roc_auc)

        preds = []
        for image_id, bbox in localization.items():
            for probability, x, y in bbox[:, :3]:
                pred = Prediction(image_id, probability, np.array([x, y]))
                preds.append(pred)

        froc = compute_froc(self.localization_gt, preds)

        return {'score': roc_auc, 'AUC': roc_auc, 'FROC': froc, 'coco': coco_metric}


def create_dataset(data_root, mode, data_fold=0):
    """
    data_root:
        - train/*.jpg
        - dev/*.jpg
        - train.json
        - train.csv
        - dev.json
        - dev.csv
    """
    data_root = Path(data_root)

    if data_fold == 0:
        if mode == "train":
             image_dir = "train"
        elif mode in ('val', 'test'):
             image_dir = "dev"
        image_dir = data_root / image_dir
        dataset = Dataset(image_dir, image_dir.with_suffix('.json'), image_dir.with_suffix('.csv'))
    else:
        train_dataset = create_dataset(data_root, "train")
        dev_dataset = create_dataset(data_root, "val")
        k = len(dev_dataset) * data_fold
        test_image_ids = [train_dataset.ids[i + k] for i in range(len(dev_dataset))]
        if mode == 'train':
            train_dataset.exclude(test_image_ids)
            dataset = train_dataset + dev_dataset
        else:
            train_dataset.include(test_image_ids)
            dataset = train_dataset
            assert len(dataset.classification_gt) == len(dataset.ids)

    # if mode in ('val', 'test'):
    #     dataset = expand_dataset_by_flips(dataset, hflip=True, vflip=True, dflip=True)

    return dataset


def add_dataset_argument(parser):
    group = parser.add_argument_group('options of dataset')
    group.add_argument('--data-root', default='/media/data/object_cxr', type=str, help='path to dataset')
    group.add_argument('--data-fold', default=1, type=int, help='data fold')
    return group
