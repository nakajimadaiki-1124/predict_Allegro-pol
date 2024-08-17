Copyright (c) 2023 The President and Fellows of Harvard College. All rights reserved.

# allegro-pol

`allegro-pol` is an extension package of the `nequip` framework that adapts the Allegro architecture (another `nequip` extension package)
for the prediction of polarization and its related properties.

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
