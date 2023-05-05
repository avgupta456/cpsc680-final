#!/bin/bash


TIMEFORMAT='It took %R seconds'
time {
    echo "Training aware vanilla label predictor model"
    python -m src.vanilla.train --dataset=$1 --seed=$2
}

echo "-----------------------------------------------------"

echo "Eval Vanilla"
python -m src.eval.main --dataset=$1
