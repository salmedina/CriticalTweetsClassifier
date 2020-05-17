#!/bin/zsh

# ../data/experiments/*.yaml
DESCRIPTOR_PATH=$1

# Manually set
TASK=adversarial    # {criticality, multitask, adversarial}
EMBEDDING_TYPE=bert # {glove, bert}
LEARNING_RATE=0.01
WEIGHT_DECAY=0.000277
MOMENTUM=0.025804
DROPOUT=0.0
NUM_EPOCHS=40

python3 -u train.py --task=$TASK \
--embedding_type=$EMBEDDING_TYPE \
--num_epochs=$NUM_EPOCHS \
--exp_desc=$DESCRIPTOR_PATH \
--lr=$LEARNING_RATE \
--wd=$WEIGHT_DECAY \
--momentum=$MOMENTUM \
--dropout=$DROPOUT \
--mute
