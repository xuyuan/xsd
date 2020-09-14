from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from detnet.trainer.data import Dataset
from detnet.trainer.data.image_folder import get_image_files


class TestDataset(Dataset):
    def __init__(self, input_file):
        input_file = Path(input_file)
        if input_file.is_file():
            self.root = input_file.parent
            df = pd.read_csv(input_file, names=['file_name'])
            self.ids = df['file_name'].to_list()
        elif input_file.is_dir():
            self.root = input_file
            self.ids = get_image_files(self.root)
        else:
            raise RuntimeError(input_file)

    def __len__(self):
        return len(self.ids)

    def getitem(self, idx):
        file_name = self.ids[idx]
        image = Image.open(self.root / file_name)
        return dict(image_id=file_name, input=image)

    def prediction_to_classification_and_localization(self, predictions,
                                                      output_classification_prediction_csv_path,
                                                      output_localization_prediction_csv_path,
                                                      scale_prediction=True):
        classification = {}
        localization = {}
        for sample in self:
            image_id = sample['image_id']
            pred = predictions[str(image_id)]
            bbox = pred[0]
            if len(bbox) == 0:
                classification[image_id] = 0
                localization[image_id] = bbox
            else:
                classification[image_id] = max(bbox[:, 0])
                if scale_prediction:
                    img = sample['input']
                    width, height = img.width, img.height
                    scale = np.asarray([1, width, height, width, height])
                    bbox = bbox * scale
                localization[image_id] = bbox

        with open(output_classification_prediction_csv_path, 'w') as output:
            output.write("image_path,prediction\n")
            for i in self.ids:
                p = classification[i]
                output.write(i + ',' + ('%.5f' % p) + '\n')

        with open(output_localization_prediction_csv_path, 'w') as output:
            output.write("image_path,prediction\n")
            for i in self.ids:
                bboxes = localization[i]
                loc = []
                for bbox in bboxes:
                    l = ['%.3f' % bbox[0]] + [str(int(v)) for v in bbox[1:3]]
                    loc.append(' '.join(l))
                loc_str = ';'.join(loc)
                output.write(i + ',' + loc_str + '\n')

    def load_prediction(self, predictions):
        """return list of dict
            {
            "image_id" : str,
            "category_id" : int,
            "bbox" : [ x, y, width, height ],
            "score" : float
            }
        """
        results = []
        for sample in self:
            image_id = sample['image_id']
            det = predictions[str(image_id)]
            img = sample['input']
            width, height = img.width, img.height
            scale = np.asarray([1, width, height, width, height])

            for cls, bbox in enumerate(det):
                bbox = bbox * scale  # [conf, cx, cy, w, h]
                bbox[:, 1:3] -= (bbox[:, 3:5] / 2)
                for box in bbox:
                    score = float(box[0])
                    bbox_float = [float(v) for v in box[1:5]]
                    results.append(dict(image_id=image_id, category_id=1, bbox=bbox_float, score=score))
        return results


if __name__ == "__main__":
    dataset = TestDataset('/home/xu/Downloads/object_cxr/test/image_path.csv')
    print(dataset)
    #print(dataset[0])
    prediction = {'valid_image/00001.jpg': [[]],
                  'valid_image/00002.jpg': [np.ones((3, 5))],
                  'valid_image/00003.jpg': [[]],
                  'valid_image/00004.jpg': [[]],
                  'valid_image/00005.jpg': [[]],
                  'valid_image/00006.jpg': [[]],
                  'valid_image/00007.jpg': [[]],
                  'valid_image/00008.jpg': [[]],
                  }
    dataset.prediction_to_classification_and_localization(prediction, '/tmp/cls.csv', '/tmp/loc.csv')
