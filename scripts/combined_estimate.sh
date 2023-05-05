#!/bin/bash

TIMEFORMAT='It took %R seconds'
time {
    echo "Training vanilla label predictor model"
    python -m src.vanilla.train --dataset=$1 --seed=$2

    echo "Training aware vanilla label predictor model"
    python -m src.vanilla.train --dataset=$1_aware --seed=$2

    echo "Training vanilla sensitive attribute classifier model"
    python -m src.vanilla.train --dataset=$1 --target_name=sens_attr --seed=$2

    echo "Debiasing Nodes on Vanilla"
    python -m src.node.train --estimate_sens_attrs --dataset=$1 --seed=$2

    echo "Training debiased node label predictor model"
    python -m src.vanilla.train --dataset=$1_node --seed=$2

    echo "Debiasing Edges on Vanilla (Combined)"
    python -m src.combined --dataset=$1 --seed=$2

    echo "Training debiased edge label predictor model"
    python -m src.vanilla.train --dataset=$1_edge --seed=$2
}

echo "-----------------------------------------------------"

echo "Eval Debiased Edges (Combined)"
python -m src.eval.main --dataset=$1_edge
