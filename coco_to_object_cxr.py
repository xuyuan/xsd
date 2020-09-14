
if __name__ == '__main__':
    import argparse
    import json
    import numpy as np
    from data.test_dataset import TestDataset
    from detnet.ensemble import convert_submission

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prediction", type=str, help='path to prediction.json')
    parser.add_argument("-i", "--input", type=str, help='path to input image_path.csv')
    parser.add_argument('--output-classification-prediction-csv-path', type=str, help='path of export file')
    parser.add_argument('--output-localization-prediction-csv-path', type=str, help='path of export file')
    args = parser.parse_args()

    dataset = TestDataset(args.input)

    data = json.load(open(args.prediction))
    detections = convert_submission(data)

    predictions = {}
    # fill empty detection
    for image_id in dataset.ids:
        bbox = np.asarray(detections[image_id][1])
        if bbox.size:
            bbox[:, 1:3] += bbox[:, 3:5] * 0.5  # center
        predictions[image_id] = [bbox]

    dataset.prediction_to_classification_and_localization(predictions,
                                                          args.output_classification_prediction_csv_path,
                                                          args.output_localization_prediction_csv_path,
                                                          scale_prediction=False)