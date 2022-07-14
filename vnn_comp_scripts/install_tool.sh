#!/bin/bash

set -e

VERSION_STRING=v1
INSTALL_USER=ubuntu

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

if [ $(id -u) = 0 ]; then
   echo "The script must not be run as root."
   exit 1
fi

if [[ ! $CONDA_DEFAULT_ENV == "prima4complete" ]]; then
  eval "$(conda shell.bash hook)"
  conda create --name prima4complete python=3.7 -y
  conda init bash
  conda activate prima4complete
  echo "created conda environment $CONDA_DEFAULT_ENV"
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd "$SCRIPT_DIR/.."
echo | sudo -H -E bash "setup.sh"

echo "Installing gurobi as $(whoami)"
cd gurobi912/linux64/
if [[ $CONDA_DEFAULT_ENV == "" ]]; then
  python3 setup.py install
else
  ~/anaconda3/envs/$CONDA_DEFAULT_ENV/bin/python3 setup.py install
fi
cd ../../

echo "Installing requirements as $(whoami)"
if [[ $CONDA_DEFAULT_ENV == "" ]]; then
  python3 -m pip install -r requirements.txt
else
  ~/anaconda3/envs/$CONDA_DEFAULT_ENV/bin/python3 -m pip install -r requirements.txt
fi

# add current directory to pythonpath
export PYTHONPATH=$PYTHONPATH:$PWD

cd $SCRIPT_DIR

echo | chmod 777 gurobi_install.sh

