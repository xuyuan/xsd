from PIL.ImageFilter import *
from torchvision.transforms import ColorJitter

from detnet.trainer.transforms import *
from detnet.trainer.transforms.vision import *
from detnet.trainer.transforms.auto_aug import *


class XMixUp(MixUp):
    def transform_bbox(self, bbox):
        src_bbox = self.copy_sample['bbox']
        return np.vstack([bbox, src_bbox])


class TrainAugmentation(Compose):
    def __init__(self, size, bg_color='black',
                 crop_remove_small_box=True,
                 truncate_box=True,
                 random_pad_max_ratio=0.0,
                 cut_out=True,
                 min_scale=0.9,
                 max_scale=1.1,
                 cut_mix_dataset=None,
                 random_rotate=5,
                 auto_aug=0,
                 rand_aug=0):
        BG_COLORS = dict(black=(0, 0, 0), white=(255, 255, 255), gray=TORCH_VISION_MEAN * 255)
        bg_color = BG_COLORS[bg_color]

        transforms = []
        if random_pad_max_ratio > 0:
            max_padding = tuple(s * random_pad_max_ratio for s in size)
            transforms += [RandomApply([RandomPad(max_padding, bg_color)])]

        transforms += [TryApply(RandomCrop(min_size=np.asarray(size) * min_scale,
                                           max_size=np.asarray(size) * max_scale,
                                           remove_bbox_outside=crop_remove_small_box,
                                           truncate_bbox=truncate_box,
                                           focus=False),
                                max_trail=5, fallback_origin=True)]

        transforms.append(Resize(size))

        transforms += [RandomApply([HorizontalFlip()]),
                       RandomApply([VerticalFlip()]),
                       RandomApply([Transpose()])]
        #
        if cut_mix_dataset:
            transforms += [RandomApply(RandomCopyPaste(1/4, 1/2, cut_mix_dataset))]

        if cut_out:
            cut_out = [RandomApply(TryApply(CutOut(max_size=max(size) // 4, fill=i))) for i in (None, bg_color)]
            transforms += cut_out

        color_jitter = RandomChoice([ColorJitter(brightness=0.25, contrast=0.5),
                                     AutoContrast(),
                                     Equalize(),
                                     CLAHE()
                                     ])

        noise = RandomChoice([GaussNoise(0.1),
                              SaltAndPepper(),
                              RandomSharpness(),
                              JpegCompression(),
                              RandomChoice([ImageFilter(BLUR),
                                            ImageFilter(DETAIL),
                                            ImageFilter(ModeFilter(size=3)),
                                            ImageFilter(GaussianBlur()),
                                            ImageFilter(MaxFilter(size=3)),
                                            ImageFilter(MedianFilter(size=3))])])

        # deform
        deform = RandomChoice([RandomHorizontalShear(0.3),
                               RandomSkew(0.3),
            RandomRotate(random_rotate),
            GridDistortion(),
            #ElasticDeformation(approximate=True),
        ])

        transforms.append(RandomApply([RandomChoice([
            color_jitter,
            noise,
            deform,
            COCOAugment(),
        ])]))

        if auto_aug > 0:
            transforms.append(RandomApply(ImageNetAugment(), p=auto_aug))
        if rand_aug > 0:
            transforms.append(RandAug(rand_aug, 0.5))

        super().__init__(transforms)