### Cloning
This repository contains a submodule. Please make sure that you have access rights to the submodule repository for cloning. After that either clone recursively via 

```
git clone --recurse-submodules https://github.com/mnmueller/mn_bab_vnn_2022.git
```

or clone normally and initialize the submodule later on

```
git clone https://github.com/mnmueller/mn_bab_vnn_2022.git
git submodule init
git submodule update
```

There's no need for a further installation of the submodules.

In case git lfs has not been initialized/installed, follow the official [instructions](https://git-lfs.github.com/) and afterwards pull the corresponding files from the submodule via

```
cd vnn-comp-2022-sup/
git lfs pull
```

### Installation
This script installs a few necessary libraries, the ELINA library, clones the repo and installs the necessary dependencies. It was tested on a AWS Deep Learning AMI (Ubuntu 18.04) instance. (Note that this requires sudoer rights)

```
source setup.sh
```

### Example usage

```
python src/verify.py -c configs/cifar10_conv_small.json
```

### Developer setup
First install the necessary dependencies (you might want to create a [virtual environment](https://docs.python.org/3/library/venv.html) first).
```
pip install -r requirements.txt
pip uninstall onnx2pytorch # in submodule
```

In order to enforce some minimal coding guidelines, there are a few pre-commit hooks. These are checks that need to succeed before a commit can be made. In order to enable them, one should run the following:
```
pre-commit install
```

To run the tests, you will need a valid gurobi license, for example at:
~/anaconda3/envs/prima4complete/lib/python3.7/site-packages/gurobipy/.libs/gurobi.lic

If it expired, you might have to update dependencies to a more current version of gurobipy.
