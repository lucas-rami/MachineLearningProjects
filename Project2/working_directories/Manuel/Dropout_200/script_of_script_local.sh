#bin/bash

for i in `seq 1 9`
do
  echo "Dropout: 0.${i}"
  sed "s/Dropout = 0/Dropout = 0.${i}/g" UNET_patch_120_template.py > UNET_patch_120_Dropout_0.${i}.py
  sed "s/python3/python3 testCNN_200_Dropout_0.${i}.py/g" test_script_template.sh > test_script_0.${i}.sh
done
