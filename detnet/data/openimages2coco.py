import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
from PIL import Image
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Open Images annotations into MS Coco format')
    parser.add_argument('-i', '--input', required=True, type=str, help="annotation bbox csv file, e.g. train-annotations-bbox.csv")
    parser.add_argument('--image-dir', type=str, help="path to image folder")
    parser.add_argument('--class-descriptions', required=True, type=str, help="class-descriptions-boxable.csv")
    parser.add_argument('-o', '--output', required=True, type=str, help="coco annotation json")
    parser.add_argument('--classes', type=str, help="classnames")
    args = parser.parse_args()

    classnames = args.classes.split(',') if args.classes else None
    cls_id = pd.read_csv(args.class_descriptions, names=['cls', 'name'], index_col=1).to_dict()['cls']
    if not classnames:
        classnames = list(cls_id.keys())
    category_ids = {c: i+1 for i, c in enumerate(classnames)}
    categories = [{'supercategory': 'none', 'id': v, 'name': k} for k, v in category_ids.items()]  # TODO: supercategory
    category_ids = {cls_id[k]: v for k, v in category_ids.items()}


    df = pd.read_csv(args.input)
    image_annotations = defaultdict(list)
    for idx, anno in df.iterrows():
        image_id = anno['ImageID']

        xmin = anno['XMin']
        ymin = anno['YMin']
        o_width = anno['XMax'] - xmin
        o_height = anno['YMax'] - ymin

        category_id = category_ids[anno['LabelName']]

        image_annotations[image_id].append({
            'id': idx,
            'image_id': image_id,
            'iscrowd': 0,
            'bbox': [xmin, ymin, o_width, o_height],
            'category_id': category_id,
            'ignore': 0,
            'segmentation': []  # This script is not for segmentation
            })

    # images
    images = []
    annotations = []
    image_root = Path(args.image_dir) if args.image_dir else None
    if image_root:
        for image_id in image_annotations:
            file_path = (image_root / image_id).with_suffix('.jpg')
            if not file_path.exists():
                print(file_path, 'not exists')
                continue
            image = Image.open(file_path)
            width = image.width
            height = image.height
            images.append({'id': image_id, 'file_name': file_path.name, 'width': width, 'height': height})
            for anno in image_annotations[image_id]:
                xmin, ymin, o_width, o_height = anno['bbox']
                xmin *= width
                ymin *= height
                o_width *= width
                o_height *= height
                anno['bbox'] = [int(xmin), int(ymin), int(o_width), int(o_height)]
                anno['area'] = int(o_width) * int(o_height)
                annotations.append(anno)

    print('categories:', categories)
    print('images:', len(images))
    print('annotations:', len(annotations))

    with open(args.output, 'w') as f:
        json.dump(dict(categories=categories, images=images, annotations=annotations), f)
