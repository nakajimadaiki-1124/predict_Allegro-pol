Copyright (c) 2023 The President and Fellows of Harvard College. All rights reserved.

# allegro-pol

`allegro-pol` is an extension package of the `nequip` framework that adapts the Allegro architecture (another `nequip` extension package)
for the prediction of the electric response of materials (polarization, Born charges, polarizability) in addition to energy and forces within a single ML model. The ideas are described in [in this paper](https://arxiv.org/abs/2403.17207). 

## Installation

This installation requires `nequip==0.6.1` and `allegro-v2`. 

It is strongly recommended to create a fresh virtual environment. For example,
```
conda create -n allegro-pol python=3.11
conda activate allegro-pol
```

It is then recommended to install dependencies starting with `torch` first and then `nequip==0.6.1`.
```
pip install torch
pip install nequip==0.6.1
```

Then install `allegro-v2`.
```
git clone https://github.com/mir-group/allegro-v2.git
cd allegro-v2
pip install -e .
```

Finally install `allegro-pol`.
```
git clone https://github.com/mir-group/allegro-pol.git
cd allegro-pol
pip install -e .
```

## Example

BaTiO3 data and an associated config file are provided for training, which is all based on the `nequip` framework with minor extensions. 
The data is located at `data/BaTiO3.xyz` and the example config is located at `configs/BaTiO3.yaml`.
```
nequip-train configs/BaTiO3.yaml
```

## Cite

If you use this code in your own work, please cite [our work](https://arxiv.org/abs/2403.17207):
```
@article{falletta2024arxiv,
      title={Unified Differentiable Learning of Electric Response}, 
      author={Stefano Falletta and Andrea Cepellotti and Anders Johansson and Chuin Wei Tan and Albert Musaelian and Cameron J. Owen and Boris Kozinsky},
      journal={arXiv:2403.17207},
      year={2024},
      url={https://arxiv.org/abs/2403.17207}, 
}
```
