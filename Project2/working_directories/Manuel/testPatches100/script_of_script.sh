#bin/bash
module purge
module load intel python/3.6.5
virtualenv -p python3 --system-site-packages x86_E5v4_Mellanox_gcc_mvapich
source x86_E5v4_Mellanox_gcc_mvapich/bin/activate
pip install --upgrade pip setuptools
pip install --no-cache-dir --upgrade tensorflow
pip install --no-cache-dir --upgrade keras
pip install --no-cache-dir --upgrade matplotlib
pip install --no-cache-dir --upgrade numpy
pip install --no-cache-dir --upgrade matplotlib
pip install --no-cache-dir --upgrade scipy
pip install --no-cache-dir --upgrade Pillow

sbatch test_script.sh
