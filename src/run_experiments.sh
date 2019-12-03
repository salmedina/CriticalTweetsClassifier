#!/bin/zsh

DESCRIPTOR_PATH=$1

echo "*** Baseline GLOVE ***"
python3 -u train.py --task=criticality --embedding_type=glove --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=0.01
echo "*** Baseline BERT ***"
python3 -u train.py --task=criticality --embedding_type=bert --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=0.01
echo "*** MULTITASK GLOVE***"
python3 -u train.py --task=multitask --embedding_type=glove --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=0.01
echo "*** MULTITASK BERT ***"
python3 -u train.py --task=multitask --embedding_type=bert --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=0.01
echo "*** ADVERSARIAL GLOVE ***"
python3 -u train.py --task=adversarial --embedding_type=glove --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=0.01
echo "*** ADVERSARIAL BERT ***"
python3 -u train.py --task=adversarial --embedding_type=bert --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=0.01