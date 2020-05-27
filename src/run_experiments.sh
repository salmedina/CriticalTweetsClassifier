#!/bin/zsh

DESCRIPTOR_PATH=$1
LEARNING_RATE=0.01
WEIGHT_DECAY=0.000277
MOMENTUM=0.025804
DROPOUT=0.0

echo "*** Baseline GLOVE ***"
python3 -u train.py --task=criticality --embedding_type=glove --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=$LEARNING_RATE --wd=$WEIGHT_DECAY --momentum=$MOMENTUM --dropout=$DROPOUT --mute
echo "*** Baseline BERT ***"
python3 -u train.py --task=criticality --embedding_type=bert --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=$LEARNING_RATE --wd=$WEIGHT_DECAY --momentum=$MOMENTUM --dropout=$DROPOUT --mute
echo "*** MULTITASK GLOVE***"
python3 -u train.py --task=multitask --embedding_type=glove --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=$LEARNING_RATE --wd=$WEIGHT_DECAY --momentum=$MOMENTUM --dropout=$DROPOUT --mute
echo "*** MULTITASK BERT ***"
python3 -u train.py --task=multitask --embedding_type=bert --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=$LEARNING_RATE --wd=$WEIGHT_DECAY --momentum=$MOMENTUM --dropout=$DROPOUT --mute
echo "*** ADVERSARIAL GLOVE ***"
python3 -u train.py --task=adversarial --embedding_type=glove --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=$LEARNING_RATE --wd=$WEIGHT_DECAY --momentum=$MOMENTUM --dropout=$DROPOUT --mute
echo "*** ADVERSARIAL BERT ***"
python3 -u train.py --task=adversarial --embedding_type=bert --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=$LEARNING_RATE --wd=$WEIGHT_DECAY --momentum=$MOMENTUM --dropout=$DROPOUT --mute
