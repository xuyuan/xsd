#!/bin/bash

source $(conda info --root)/etc/profile.d/conda.sh
conda activate bottle_defects_detection

N_GPUS=$(nvidia-smi -L | wc -l)
TESTDATA=/tcdata/testA/images

TMP=/dev/shm

#CAP_MODEL='weights/cap_11201_best_metric.model.pth'
CAP_MODEL='weights/cap_11248_swa.model.pth'
WINE_MODEL='weights/wine_11205_best_metric.model.pth'
#BOTTLE_MODEL='weights/bottle_11202_best_metric.model.pth'
BOTTLE_MODEL='weights/bottle_11266_swa.model.pth'

python inference.py -i $TESTDATA --category=cap --image-size=492x492 -m $CAP_MODEL --export cap_submission.json --batch-size=8
python inference.py -i $TESTDATA --category=wine --image-size=1536x2097 -m $WINE_MODEL --batch-size=4 --export wine_submission.json
python inference.py -i $TESTDATA --category=bottle --image-size=800x800 -m $BOTTLE_MODEL --export bottle_submission.json --tta=orig,hflip,vflip
#python merge_submission.py -o result.json
#cp result.json /result.json
#zip -r submission.zip submission

