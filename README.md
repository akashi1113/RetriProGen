# RetriProGen

## Title
RetriProGen: A Retrieval-Guided Molecular Generation Model with Protein-Ligand Joint Representation for Property-Aware Drug Design

## Abstract
Drug discovery is a complex multi-objective optimization problem that requires balancing molecular interaction with target proteins, favorable physicochemical properties, and synthetic accessibility. Identifying and optimizing molecular structures that can effectively bind to specific protein pockets is a key challenge. Traditional drug discovery methods often rely on extensive screening, which is time-consuming and inefficient. Recent advances in deep-generative models have revolutionized molecular design by enabling efficient navigation of a vast chemical space. 
To address these issues, we propose a novel generative model based on protein-molecule representation for targeted molecular design named RetriProGen. 
![image](model.png)

## Setup
Please install RetriProGen in a virtual environment to ensure it has conflicting dependencies.
```
Python == 3.8
PyTorch == 1.13.1
scikit-learn == 1.5.0
pandas == 2.2.1
numpy == 1.22.4
RDKit == 2023.9.6
transformers == 4.41.2
PyG == 2.5.3
Install pytorch_geometric following instruction at https://github.com/rusty1s/pytorch_geometric
```
## Dataset
In this study, we employ the CrossDockedPocket10 dataset, which is introduced by [1].

## Access data

The required data files and MolT5 model are available via Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15266318.svg)](https://doi.org/10.5281/zenodo.15266318)

Dataset:
Download the data package and place all files in the data/ directory.

MolT5 Model:
Download the pre-trained MolT5 model and place it in the models/molT5/ directory.

These files must be properly installed before running the training scripts.

## Run the model
Firstly, you will use the `python process_data.py` script to process raw data, then use `python preprocess_database.py` and `python get_retrieval_database.py` to construct the retrieval database.

Secondly, to train the model, you will use the `python train_ret.py` script. This script accepts several command-line arguments to customize the training process.

Then, you will use the `python gen.py` scrip to generate molecules using the trained model.

We also provide a pre-trained model `model.pt`, which you can directly apply in the `gen.py` script and then run the script to generate molecules.

The model is also available via Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15266318.svg)](https://doi.org/10.5281/zenodo.15266318)

## Reference

[1] Luo S, Guan J, Ma J, et al. A 3D generative model for structure-based drug design[J]. Advances in Neural Information Processing Systems, 2021, 34: 6229-6239.

