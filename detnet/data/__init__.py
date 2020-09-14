from pathlib import Path
import json
from ..trainer.data import TransformedDataset, Subset, ImageFolder, RollSplitSet
from .coco import COCODetection


def load_folds(split_file=None):
    if split_file is None:
        split_file = Path(__file__).parent / 'stratified_10_fold.json'
    else:
        split_file = Path(split_file)
    return json.load(split_file.open())


def create_dataset(data_root, mode, data_fold=0, split_file=None, transform=None, pseudolabel_rate=0):
    data_root = Path(data_root)
    train = mode == 'train'

    image_ids = None
    if data_fold >= 0:
        data_folds = load_folds(split_file=split_file)
        if train:
            image_ids = sum([fold for i, fold in enumerate(data_folds) if i != data_fold], [])
        else:
            image_ids = data_folds[data_fold]

    dataset = COCODetection(data_root / 'images', data_root / 'annotations.json', image_ids=image_ids)

    if pseudolabel_rate > 0:
        raise NotImplementedError
        # testA = data_root.parent / 'chongqing1_round1_testA_20191223'
        # testB = data_root.parent / 'chongqing1_round1_testB_20200210'
        # testA = DatasetClass(testA / 'images', testA / 'annotations.json', train=train)
        # teatB = DatasetClass(testB / 'images', testB / 'annotations.json', train=train)
        # pseudolabel = testA + testB
        # splits = int(len(pseudolabel) / len(dataset) / pseudolabel_rate)
        # pseudolabel = RollSplitSet(pseudolabel, splits)
        # dataset += pseudolabel

    if transform:
        dataset = TransformedDataset(dataset, transform=transform)

    return dataset


def add_dataset_argument(parser):
    group = parser.add_argument_group('options of dataset')
    group.add_argument('--data-root', default='/media/data/coco', type=str, help='path to dataset')
    group.add_argument('--data-split-file', type=str, help='json file of train/valid data splits')
    group.add_argument('--data-fold', default=0, type=int, help='data fold id for cross validation')
    group.add_argument('--data-pseudolabel-rate', default=0, type=float, help="percentage of pseudolabel in train dataset")
    return group

