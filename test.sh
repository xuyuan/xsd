#!/bin/bash

#cl run image_path.csv:image_path.csv valid_image:valid_image object_cxr:object_cxr test.sh:test.sh 12946-best_metric.model.pth:_12946-best_metric.model.pth 12956-best_metric.model.pth:_12956-best_metric.model.pth "bash ./test.sh image_path.csv predictions_classification.csv predictions_localization.csv" -n run-predictions --request-docker xuyuancn/object_cxr:latest --request-gpus 1 --request-memory 8g

source $(conda info --root)/etc/profile.d/conda.sh
conda activate object_cxr

N_GPUS=$(nvidia-smi -L | wc -l)

mkdir results
for model_id in 12946 12956 13056
do
  for tta in orig hflip x1.2 x1.4
  do
    python object_cxr/inference.py --batch-size=1 --resize 1024 -m ${model_id}-best_metric.model.pth --tta ${tta} -i $1 --export results/${model_id}_${tta}.json --output-classification-prediction-csv-path cls.csv --output-localization-prediction-csv-path loc.csv
  done
done

#python -m object_cxr.detnet.ensemble results -o ensemble_nms.json -m nms
python -m object_cxr.detnet.ensemble results -o ensemble.json

#python object_cxr/coco_to_object_cxr.py -i $1 --output-classification-prediction-csv-path $2 --output-localization-prediction-csv-path loc.csv ensemble_nms.json
python object_cxr/coco_to_object_cxr.py -i $1 --output-classification-prediction-csv-path $2 --output-localization-prediction-csv-path $3 ensemble.json