#!/bin/zsh

DESCRIPTOR_PATH=$1
LEARNING_RATE=0.086189
WEIGHT_DECAY=0.012114
MOMENTUM=0.687299

echo "*** Baseline GLOVE ***"
python3 -u train.py --task=criticality --embedding_type=glove --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=$LEARNING_RATE --wd=$WEIGHT_DECAY --momentum=$MOMENTUM
echo "*** Baseline BERT ***"
python3 -u train.py --task=criticality --embedding_type=bert --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=$LEARNING_RATE --wd=$WEIGHT_DECAY --momentum=$MOMENTUM
echo "*** MULTITASK GLOVE***"
python3 -u train.py --task=multitask --embedding_type=glove --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=$LEARNING_RATE --wd=$WEIGHT_DECAY --momentum=$MOMENTUM
echo "*** MULTITASK BERT ***"
python3 -u train.py --task=multitask --embedding_type=bert --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=$LEARNING_RATE --wd=$WEIGHT_DECAY --momentum=$MOMENTUM
echo "*** ADVERSARIAL GLOVE ***"
python3 -u train.py --task=adversarial --embedding_type=glove --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=$LEARNING_RATE --wd=$WEIGHT_DECAY --momentum=$MOMENTUM
echo "*** ADVERSARIAL BERT ***"
python3 -u train.py --task=adversarial --embedding_type=bert --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=$LEARNING_RATE --wd=$WEIGHT_DECAY --momentum=$MOMENTUM