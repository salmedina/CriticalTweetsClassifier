#!/bin/zsh

# ../data/experiments/*.yaml
EVENT_NAME=$1
SPLIT_ID=$2
DESCRIPTOR_PATH="../data/experiments/${EVENT_NAME}_${SPLIT_ID}.yaml"

# Manually set
TASK=multitask    # {criticality, multitask, adversarial}
EMBEDDING_TYPE=bert # {glove, bert}
LEARNING_RATE=0.01
WEIGHT_DECAY=0.000277
MOMENTUM=0.025804
DROPOUT=0.0
NUM_EPOCHS=40
OUTPUT_PATH="../output/models/${EVENT_NAME}_${SPLIT_ID}_${TASK}.pth"

python3 -u train.py --task=$TASK \
--embedding_type=$EMBEDDING_TYPE \
--num_epochs=$NUM_EPOCHS \
--exp_desc=$DESCRIPTOR_PATH \
--lr=$LEARNING_RATE \
--wd=$WEIGHT_DECAY \
--momentum=$MOMENTUM \
--dropout=$DROPOUT \
--output_path=$OUTPUT_PATH \
--mute
