
save_path=data/benchmark_esa_binarized/raw_predictions/
data_path=data/ESA-ADB/binarized/multivariate/windowed/ESA_ADB_512/

#srun --nodes=1 --cluster=hpda2 --partition=hpda2_compute_gpu --gres=gpu:1 --time=00:30:00  --mem=99000 --cpus-per-task=48 \
#     nohup python eval_deep_model.py --model=convnet --model_path=results/weights/supervised/convnet_default_512/model_30012023_173428 \
#     --params=models/configuration/convnet_default.json \
#     --path_save=$save_path \
#     --data=$data_path\
#     >> ms_convnet_512.out &



#!/bin/bash


# Evaluate Inception Time

srun --nodes=1 --cluster=hpda2 --partition=hpda2_compute_gpu --gres=gpu:1 --time=00:10:00  --mem=99000 --cpus-per-task=48 \
     python3 eval_deep_model.py --model=inception_time --model_path=results/weights/supervised/inception_time_default_512 \
     --params=models/configuration/inception_time_default.json \
     --path_save=$save_path \
     --data=$data_path >> inception.out &

# Evaluate Resnet

srun --nodes=1 --cluster=hpda2 --partition=hpda2_compute_gpu --gres=gpu:1 --time=00:10:00  --mem=99000 --cpus-per-task=48 \
     python3 eval_deep_model.py --model=resnet --model_path=results/weights/supervised/resnet_default_512 \
     --params=models/configuration/resnet_default.json --path_save=$save_path --data=$data_path >> resnet.out &

# Evaluate Signal Transformer (SiT) with Convolutional Patch

srun --nodes=1 --cluster=hpda2 --partition=hpda2_compute_gpu --gres=gpu:1 --time=00:10:00  --mem=99000 --cpus-per-task=48 \
     python3 eval_deep_model.py --model=sit --model_path=results/weights/supervised/sit_conv_patch_512 \
     --params=models/configuration/sit_conv_patch.json --path_save=$save_path --data=$data_path >> sit_conv_patch.out &

# Evaluate Signal Transformer (SiT) with Linear Patch

srun --nodes=1 --cluster=hpda2 --partition=hpda2_compute_gpu --gres=gpu:1 --time=00:10:00  --mem=99000 --cpus-per-task=48 \
     python3 eval_deep_model.py --model=sit --model_path=results/weights/supervised/sit_linear_patch_512 \
     --params=models/configuration/sit_linear_patch.json --path_save=$save_path --data=$data_path >> sit_linear_patch.out &


# Evaluate Signal Transformer (SiT) with Original Stem

srun --nodes=1 --cluster=hpda2 --partition=hpda2_compute_gpu --gres=gpu:1 --time=00:10:00  --mem=99000 --cpus-per-task=48 \
     python3 eval_deep_model.py --model=sit --model_path=results/weights/supervised/sit_stem_original_512 \
     --params=models/configuration/sit_stem_original.json --path_save=$save_path --data=$data_path >> sit_original_stem.out &

# Evaluate Signal Transformer (SiT) with ReLU Stem

srun --nodes=1 --cluster=hpda2 --partition=hpda2_compute_gpu --gres=gpu:1 --time=00:10:00  --mem=99000 --cpus-per-task=48 \
     python3 eval_deep_model.py --model=sit --model_path=results/weights/supervised/sit_stem_relu_512 \
     --params=models/configuration/sit_stem_relu.json --path_save=$save_path --data=$data_path >> sit_relu_stem.out &
