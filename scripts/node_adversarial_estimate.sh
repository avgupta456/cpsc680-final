#!/bin/bash

TIMEFORMAT='It took %R seconds'
time {
    echo "Training vanilla sensitive attribute classifier model"
    python -m src.vanilla.train --dataset=$1 --target_name=sens_attr --seed=$2

    echo "Debiasing Nodes on Vanilla"
    python -m src.node.train --estimate_sens_attrs --dataset=$1 --seed=$2

    echo "Training debiased node label predictor model"
    python -m src.vanilla.train --dataset=$1_node --seed=$2
}

echo "-----------------------------------------------------"

echo "Eval Debiased Nodes"
python -m src.eval.main --dataset=$1_node
