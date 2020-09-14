"""
demo for object detection
"""
import numpy as np
import cv2
import time
import argparse
from .utils.visualization import draw_detection, draw_masks_on_image
from .trainer.data.image_stream import ImageStream
from .trainer.transforms.vision import Resize


def draw_results(image, detection, prob_threshold=0.5):
    if isinstance(detection, tuple):
        bboxes, masks = detection
    else:
        bboxes = detection
        masks = []

    image = draw_detection(image, bboxes, net.classnames, prob_threshold=prob_threshold)
    for m in masks:
        image = draw_masks_on_image(image, m)
    return image


if __name__ == '__main__':
    from .nn import load
    from .trainer.utils import choose_device

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', default='auto', choices=['cuda', 'cpu'], help='running with cpu or cuda')
    parser.add_argument("-m", "--model", type=str, default='torchvision:fasterrcnn_resnet50_fpn',
                        help='pre/trained model file')
    parser.add_argument("--threshold", type=float, default=0.5, help='threshold for accepting detection')
    parser.add_argument('-i', '--input', default=0, type=str, help='input images, video, folder, youtube url, etc.')
    parser.add_argument("--resize", type=lambda s: [int(item) for item in reversed(s.split('x'))] if 'x' in s else int(s),
                        help='resize the smaller edge of the image if positive number given, or resize to given size if 2 numbers given')
    parser.add_argument("--max-image-size", type=int, help='limit the maximum size of longer edge of image, it is useful to avoid OOM')
    parser.add_argument("--interval", type=int, help='detection interval between frames')
    args = parser.parse_args()

    net = load(args.model)
    net.float()
    net.eval()
    device = choose_device(args.device)
    print('use', device)
    net = net.to(device)

    image_stream = ImageStream(args.input, interval=args.interval)

    if args.resize:
        image_stream = image_stream >> Resize(args.resize, interpolation=Resize.BILINEAR, max_size=args.max_image_size)

    start_time = time.time()
    delay = 10
    n_frames = 0
    last_file_name = None
    for sample in image_stream:

        if last_file_name and sample['file_name'] != last_file_name:
            if cv2.waitKey(0) == 27:
                break

        image = sample['input']
        d = net.predict(image)

        # Display the resulting frame
        image_with_results = draw_results(image, d, prob_threshold=args.threshold)
        rgb = np.asarray(image_with_results)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', bgr)
        n_frames += 1
        last_file_name = sample['file_name']
        if n_frames % 100 == 0:
            print("FPS: ", n_frames / (time.time() - start_time))

        k = cv2.waitKey(delay)
        if k == 27:  # ESC to exit
            break
        elif k == 32:  # space to toggle playing
            delay = 0 if delay else 10
        elif k >= 0:  # other keys: frame by frame
            delay = 0

    # cleanup
    cv2.destroyAllWindows()
