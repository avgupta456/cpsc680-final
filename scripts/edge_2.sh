#!/bin/bash

TIMEFORMAT='It took %R seconds'
time {
    echo "Training vanilla label predictor model"
    python -m src.vanilla.train --dataset=$1 --seed=$2

    echo "Training aware vanilla label predictor model"
    python -m src.vanilla.train --dataset=$1_aware --seed=$2

    echo "Debiasing Edges on Vanilla (Method 2)"
    python -m src.method2 --dataset=$1 --seed=$2

    echo "Training debiased edge label predictor model"
    python -m src.vanilla.train --dataset=$1_edge --seed=$2
}

echo "-----------------------------------------------------"

echo "Eval Debiased Edges (Method 2)"
python -m src.eval.main --dataset=$1_edge
