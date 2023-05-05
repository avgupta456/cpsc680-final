#!/bin/bash

TIMEFORMAT='It took %R seconds'
time {
    echo "Debiasing Nodes on Vanilla"
    python -m src.node.simple --dataset=$1 --seed=$2

    echo "Training debiased node label predictor model"
    python -m src.vanilla.train --dataset=$1_node --seed=$2
}

echo "-----------------------------------------------------"

echo "Eval Debiased Nodes"
python -m src.eval.main --dataset=$1_node
