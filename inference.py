if __name__ == '__main__':
    from pathlib import Path
    from detnet.inference import ArgumentParser, inference, ImageFolder, Resize
    from data import create_dataset, add_dataset_argument, ToRGB
    from data.test_dataset import TestDataset
    from nn import SingleShotDetectorWithClassifier

    parser = ArgumentParser()
    add_dataset_argument(parser.parser)
    parser.add_argument('--output-classification-prediction-csv-path', type=str, help='path of export file')
    parser.add_argument('--output-localization-prediction-csv-path', type=str, help='path of export file')
    args = parser.parse_args(raise_warning=False)

    if args.input:
        dataset = TestDataset(args.input)
    else:
        dataset = create_dataset(args.data_root, mode='test')

    dataset = dataset >> ToRGB()

    predictions = inference(dataset, args)
    if args.input:
        dataset.prediction_to_classification_and_localization(predictions,
                                                              args.output_classification_prediction_csv_path,
                                                              args.output_localization_prediction_csv_path)
