# this script assumes that you are running on Ubuntu 18.04 and have sudo rights

# install dependencies
sudo apt-get install m4

  
wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
tar -xvf gmp-6.1.2.tar.xz
cd gmp-6.1.2
./configure --enable-cxx
make
sudo make install
cd ..
rm gmp-6.1.2.tar.xz


wget https://files.sri.inf.ethz.ch/eran/mpfr/mpfr-4.1.0.tar.xz
tar -xvf mpfr-4.1.0.tar.xz
cd mpfr-4.1.0
./configure
make
sudo make install
cd ..
rm mpfr-4.1.0.tar.xz

  
wget https://github.com/cddlib/cddlib/releases/download/0.94m/cddlib-0.94m.tar.gz
tar zxf cddlib-0.94m.tar.gz
rm cddlib-0.94m.tar.gz
cd cddlib-0.94m
./configure
make
sudo make install
cd ..

  
# download repo
# git clone https://gitlab.inf.ethz.ch/markmueller/prima4complete.git
# cd prima4complete

# setup ELINA
git clone https://github.com/eth-sri/ELINA.git
cd ELINA
./configure -use-deeppoly -use-fconv
make
sudo make install
cd ..
  

# create virtual environment
CONDA_PATH="$HOME/anaconda3/bin/conda"
echo | ${CONDA_PATH} update -n base conda
echo | ${CONDA_PATH} create -n prima4complete python=3.7
echo | ${CONDA_PATH} init bash
echo | ${CONDA_PATH} activate prima4complete

wget https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz
tar -xvf gurobi9.1.2_linux64.tar.gz
cd gurobi912/linux64/src/build
sed -ie 's/^C++FLAGS =.*$/& -fPIC/' Makefile
make
cp libgurobi_c++.a ../../lib/
cd ../../
cp lib/libgurobi91.so /usr/local/lib
python3 setup.py install
cd ../../
rm gurobi9.1.2_linux64.tar.gz

export GUROBI_HOME="$(pwd)/gurobi912/linux64"
export PATH="${PATH}:/usr/lib:${GUROBI_HOME}/bin"
export CPATH="${CPATH}:${GUROBI_HOME}/include"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib:${GUROBI_HOME}/lib


# install dependencies
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

pip install --upgrade pip
pip install -r  "$SCRIPT_DIR/requirements.txt"
  

# add current directory to pythonpath
export PYTHONPATH=$PYTHONPATH:$PWD
