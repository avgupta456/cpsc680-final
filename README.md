# CPSC 680 Final Project

## About

While graph neural networks (GNNs) have emerged as a powerful tool for analyzing graph structured data, they have the tendency to perpetuate and even amplify social biases, leading to unfair and discriminatory outcomes. GNN fairness is concerned with eliminating harmful biases and ensuring models output predictions independent of protected features. While existing methods counteract attribute and structural bias in many ways, these methods are inefficient and costly, reducing real-world applicability. Unlike previous approaches, we prioritize efficiency and focus specifically on graph preprocessing to eliminate bias.

We separate the problem into node and edge debiasing. Within node debiasing, we propose two methods to learn fair node representations. First, we use linear algebra to remove the correlation between the features and the sensitive attribute. Second, we use adversarial debiasing to simultaneously train an encoder and classifier. The encoder learns to create similar embeddings that prevent the classifier from predicting the sensitive attribute. Within edge debiasing, we focus on heuristics to reduce excess homophily by removing a subset of edges between nodes of the same protected attribute. Our first method attempts to remove edges that are not likely to exist in the graph and our second method attempts to remove edges whose existence in the graph is heavily influenced by the sensitive attribute. Finally, we propose a combined method that uses both the adversarial debiasing node method and the second edge method.

As baselines, we consider a vanilla model both aware and unaware of the sensitive attribute. We also consider existing state-of-the-art methods FairGNN and EDITS. We experiment with five datasets, ultimately eliminating two due to unique graph properties. On the German Credit, Pokec-n, and Pokec-z datasets, we evaluate our methods for runtime, performance, and fairness. We find that both the node and edge methods improve fairness significantly with little to no impact on performance. While runtime increases, it is still significantly less than FairGNN and EDITS. Although the existing baselines have greater performance on these datasets, we believe our methods fill a valuable niche as starting points for debiasing large graph datasets. We plan to continue testing our method on larger datasets not previously considered for GNN fairness and explore additional modifications to our node and edge debiasing methods.

## Usage

1. Clone repository. Copy `credit_edges.txt` from [EDITS repository](https://github.com/yushundong/EDITS/tree/main/dataset/credit) into `data/credit/credit_edges.txt` (over 100 MB).

2. Install the dependencies in the `requirements.txt` file. The main ones are `torch==2.0.0`, `torch_geometric==2.3.0`, `torchmetrics`, `pandas`, and `matplotlib`.

3. Run `./scripts/{script_name}.sh {dataset_name} {seed}` to reproduce results. You will need to manually modify the default block in `src/vanilla/argparser.py` to select `GCNConv` or `SAGEConv`.

## File Structure

- `data/` contains the raw data and PyTorch Geometric processed data files. Refer to above to load the credit edges. The `pokec` folder contains both `pokec_z` and `pokec_n`. Most of the files are copied from the FairGNN or EDITS papers.
- `models/` contains trained models. The model name directly mirrors the dataset name with optional suffixes.
- `scripts/` contains bash scripts to reproduce results. Options for `script_name` are `vanilla_unaware`, `vanilla_aware`, `node_simple`, `node_adversarial`, `edge_1`, `edge_2`, and `combined`.
- `src/` contains all the source code. `src/datasets` loads the datasets; `src/vanilla` trains the baseline models; `src/node` performs node debiasing; `src/method1.py` and `src/method2.py` perform edge debiasing; `src/combined.py` combines node and edge debiasing; and `src/eval` computes performance and fairness metrics.
