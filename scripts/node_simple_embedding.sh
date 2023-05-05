#!/bin/bash

TIMEFORMAT='It took %R seconds'
time {
    echo "Training vanilla label predictor model"
    python -m src.vanilla.train --dataset=$1_aware --seed=$2

    echo "Debiasing Nodes on Vanilla"
    python -m src.node.simple_embedding --dataset=$1 --seed=$2
}
