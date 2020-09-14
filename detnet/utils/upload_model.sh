#!/bin/bash

HASH=`sha256sum < $1 | head -c 8`
MODEL_BASE_NAME=`basename $1 .pth`
MODEL_FILE="${MODEL_BASE_NAME}-${HASH}.pth"

MODEL_ZOO="$2:../../web/model_zoo"
MODEL_ZOO_FILE="${MODEL_ZOO}/${MODEL_FILE}"

echo "$1 --> $MODEL_ZOO_FILE"
scp $1 $MODEL_ZOO_FILE

