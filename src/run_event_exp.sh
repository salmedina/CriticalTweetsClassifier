#!/bin/zsh


LEARNING_RATE=0.01
WEIGHT_DECAY=0.000277
MOMENTUM=0.025804
DROPOUT=0.0
EVENT=$1

for i in {1..$2}
do
    DESCRIPTOR_PATH="../data/experiments/${EVENT}_${i}.yaml"

    # Baseline GLOVE
    OUTPUT_PATH="../output/models/${EVENT}_${i}_base_glove.pth"
    python3 -u train.py --task=criticality --embedding_type=glove --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=$LEARNING_RATE --wd=$WEIGHT_DECAY --momentum=$MOMENTUM --dropout=$DROPOUT --output_path=$OUTPUT_PATH --mute
    # Baseline BERT
    OUTPUT_PATH="../output/models/${EVENT}_${i}_base_bert.pth"
    python3 -u train.py --task=criticality --embedding_type=bert --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=$LEARNING_RATE --wd=$WEIGHT_DECAY --momentum=$MOMENTUM --dropout=$DROPOUT --output_path=$OUTPUT_PATH --mute
    # MULTITASK GLOVE
    OUTPUT_PATH="../output/models/${EVENT}_${i}_mt_glove.pth"
    python3 -u train.py --task=multitask --embedding_type=glove --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=$LEARNING_RATE --wd=$WEIGHT_DECAY --momentum=$MOMENTUM --dropout=$DROPOUT --output_path=$OUTPUT_PATH --mute
    # MULTITASK BERT
    OUTPUT_PATH="../output/models/${EVENT}_${i}_mt_bert.pth"
    python3 -u train.py --task=multitask --embedding_type=bert --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=$LEARNING_RATE --wd=$WEIGHT_DECAY --momentum=$MOMENTUM --dropout=$DROPOUT --output_path=$OUTPUT_PATH --mute
    # ADVERSARIAL GLOVE
    OUTPUT_PATH="../output/models/${EVENT}_${i}_adv_glove.pth"
    python3 -u train.py --task=adversarial --embedding_type=glove --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=$LEARNING_RATE --wd=$WEIGHT_DECAY --momentum=$MOMENTUM --dropout=$DROPOUT --output_path=$OUTPUT_PATH --mute
    # ADVERSARIAL BERT
    OUTPUT_PATH="../output/models/${EVENT}_${i}_adv_bert.pth"
    python3 -u train.py --task=adversarial --embedding_type=bert --num_epochs=40 --exp_desc=$DESCRIPTOR_PATH --lr=$LEARNING_RATE --wd=$WEIGHT_DECAY --momentum=$MOMENTUM --dropout=$DROPOUT --output_path=$OUTPUT_PATH --mute
done