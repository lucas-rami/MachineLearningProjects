#bin/bash
module purge
module load intel python/3.6.5
virtualenv -p python3 --system-site-packages x86_E5v4_Mellanox_gcc_mvapich
source x86_E5v4_Mellanox_gcc_mvapich/bin/activate
pip install --upgrade pip setuptools
pip install --no-cache-dir --upgrade tensorflow
pip install --no-cache-dir --upgrade keras
pip install --no-cache-dir --upgrade numpy
pip install --no-cache-dir --upgrade matplotlib
pip install --no-cache-dir --upgrade scipy
pip install --no-cache-dir --upgrade Pillow

for i in `seq 1 9`
do
  echo "Dropout: 0.${i}"
  sed "s/Dropout = 0/Dropout = 0.${i}/g" UNET_patch_120_template.py > UNET_patch_120_0.${i}.py
  sed "s/python3/python3 UNET_patch_120_0.${i}.py/g" test_script_template.sh > test_script_0.${i}.sh
  sbatch test_script_0.${i}.sh
done
