from PIL import Image
from detnet.train import ArgumentParser, train
from data.data_aug import TrainAugmentation, Resize, ToRGB, XMixUp, RandomApply
# from trainer.utils import warn_with_traceback
# warnings.showwarning = warn_with_traceback


if __name__ == '__main__':
    from data import create_dataset, add_dataset_argument
    from nn import SingleShotDetectorWithClassifier

    parser = ArgumentParser()
    group = add_dataset_argument(parser)
    parser.add_argument('--data-aug-random-rotate', default=5, type=float, help='degrees of random rotation data aug')
    parser.add_argument('--data-aug-auto-aug', default=0, type=float, help='probability to apply AutoAug')
    parser.add_argument('--data-aug-rand-aug', default=0, type=int, help='the number of RandAug transformations to apply sequentially')
    args = parser.parse_args()

    #image_size = 768  #b2
    #image_size = 896  #b3
    image_size = 1024  #b4
    #image_size = 1280
    datasets = {mode: create_dataset(args.data_root, mode=mode, data_fold=args.data_fold)
                for mode in ('train', 'test')}

    data_aug = TrainAugmentation((image_size, image_size), random_rotate=args.data_aug_random_rotate,
                                 auto_aug=args.data_aug_auto_aug,
                                 rand_aug=args.data_aug_rand_aug)
    train_dataset = datasets['train'] >> Resize(image_size) >> ToRGB() >> data_aug
    datasets['train'] = train_dataset #>> RandomApply(XMixUp(train_dataset, alpha=2), p=0.3)
    datasets['test'] = datasets['test'] >> Resize(image_size, interpolation=Image.BILINEAR) >> ToRGB()

    train(datasets, args)
