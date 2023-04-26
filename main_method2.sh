echo "Training vanilla label predictor model"
python -m src.vanilla.train --dataset=$1 --seed=$2
echo "Training aware vanilla label predictor model"
python -m src.vanilla.train --dataset=$1_aware --seed=$2
echo "Training vanilla sensitive attribute classifier model"
python -m src.vanilla.train --dataset=$1 --target_name=sens_attr --seed=$2
echo "Training vanilla link prediction model"
python -m src.vanilla.train --dataset=$1_link_pred --type=edge --seed=$2

echo "-----------------------------------------------------"

echo "Debiasing Edges on Vanilla (Method 1)"
python -m src.method1 --dataset=$1 --seed=$2
echo "Training debiased edge label predictor model"
python -m src.vanilla.train --dataset=$1_edge_1 --seed=$2

echo "-----------------------------------------------------"

echo "Debiasing Edges on Vanilla (Method 2)"
python -m src.method2 --dataset=$1 --seed=$2
echo "Training debiased edge label predictor model"
python -m src.vanilla.train --dataset=$1_edge_2 --seed=$2

echo "-----------------------------------------------------"

echo "Debiasing Nodes on Vanilla"
python -m src.node.train --dataset=$1 --estimate_sens_attrs --seed=$2
echo "Training debiased node label predictor model"
python -m src.vanilla.train --dataset=$1_node --seed=$2

echo "Debiasing Aware Nodes on Vanilla"
python -m src.node.train --dataset=$1_aware --seed=$2
echo "Training aware debiased node label predictor model"
python -m src.vanilla.train --dataset=$1_aware_node --seed=$2

echo "Training debiased node link prediction model"
python -m src.vanilla.train --dataset=$1_node_link_pred --type=edge --seed=$2

echo "-----------------------------------------------------"

echo "Debiasing Edges on Debiased Nodes (Method 1)"
python -m src.method1 --dataset=$1_node --seed=$2
echo "Training debiased node + edge label predictor model (Method 1)"
python -m src.vanilla.train --dataset=$1_node_edge_1 --seed=$2

echo "-----------------------------------------------------"

echo "Debiasing Edges on Debiased Nodes (Method 2)"
python -m src.method2 --dataset=$1_node --seed=$2
echo "Training debiased node + edge label predictor model (Method 2)"
python -m src.vanilla.train --dataset=$1_node_edge_2 --seed=$2

echo "-----------------------------------------------------"

echo "Eval Vanilla"
python -m src.eval.main --dataset=$1

echo "-----------------------------------------------------"

echo "Eval Debiased Edges (Method 1)"
python -m src.eval.main --dataset=$1_edge_1

echo "-----------------------------------------------------"

echo "Eval Debiased Edges (Method 2)"
python -m src.eval.main --dataset=$1_edge_2

echo "-----------------------------------------------------"

echo "Eval Debiased Nodes"
python -m src.eval.main --dataset=$1_node

echo "-----------------------------------------------------"

echo "Eval Debiased Nodes + Edges (Method 1)"
python -m src.eval.main --dataset=$1_node_edge_1

echo "-----------------------------------------------------"

echo "Eval Debiased Nodes + Edges (Method 2)"
python -m src.eval.main --dataset=$1_node_edge_2