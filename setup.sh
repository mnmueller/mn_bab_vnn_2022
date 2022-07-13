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
conda init bash
conda create -n prima4complete python=3.7
conda activate prima4complete

  
# install dependencies
pip install -r requirements.txt
  

# add current directory to pythonpath
export PYTHONPATH=$PYTHONPATH:$PWD
