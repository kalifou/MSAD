#!/bin/bash
# Usage: ./submit_finetuning_jobs.sh

set -e


DATA_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0016/das_gkz/MSAD/data/ESA-ADB/binarized/multivariate/windowed/ESA_ADB_512/ESA-Mission1-semi-supervised/"
COMMON_ARGS="--path=$DATA_PATH --epochs=30 --batch=32 --window-size=512 --layer-wise-lr"


mkdir -p logs

submit_job() {
    JOB_NAME=$1
    LOG_FILE="logs/$2"  
    shift 2             
    PYTHON_ARGS="$@"    

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_FILE}
#SBATCH --error=${LOG_FILE}
#SBATCH --nodes=1
#SBATCH --cluster=hpda2
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mem=99000
#SBATCH --cpus-per-task=48

module purge
module load python/3.10.12-base
source .venv/bin/activate

echo "------------------------------------------------"
echo "Job '\${SLURM_JOB_NAME}' started on \$(hostname)"
echo "Date: \$(date)"
echo "------------------------------------------------"

python3 finetune_deep_model.py ${PYTHON_ARGS}

echo "------------------------------------------------"
echo "Job finished at \$(date)"
EOT
}


echo "Submitting ConvNet..."
submit_job "ft_convnet" "ft_convnet.out" \
    --model=convnet \
    --weights=results/weights/supervised/convnet_default_512/model_30012023_173428 \
    --params=models/configuration/convnet_default.json \
    --use-freezing --freeze-ratio=0.4 $COMMON_ARGS

echo "Submitting SiT Linear Patch..."
submit_job "ft_sit_lin" "ft_sit_linear_patch.out" \
    --model=sit \
    --weights=results/weights/supervised/sit_linear_patch_512/model_30012023_174048 \
    --params=models/configuration/sit_linear_patch.json \
    --use-lora --lora-rank=16 --lora-alpha=32.0 \
    --backbone-lr=1e-5 --middle-lr=5e-5 --head-lr=1e-4 $COMMON_ARGS

echo "Submitting SiT Conv Patch..."
submit_job "ft_sit_conv" "ft_sit_conv_patch.out" \
    --model=sit \
    --weights=results/weights/supervised/sit_conv_patch_512/model_30012023_174048 \
    --params=models/configuration/sit_conv_patch.json \
    --use-lora --lora-rank=16 --lora-alpha=32.0 \
    --backbone-lr=1e-5 --middle-lr=5e-5 --head-lr=1e-4 $COMMON_ARGS

echo "Submitting SiT Stem Original..."
submit_job "ft_sit_orig" "ft_sit_stem_original.out" \
    --model=sit \
    --weights=results/weights/supervised/sit_stem_original_512/model_30012023_174048 \
    --params=models/configuration/sit_stem_original.json \
    --use-lora --lora-rank=16 --lora-alpha=32.0 \
    --backbone-lr=1e-5 --middle-lr=5e-5 --head-lr=1e-4 $COMMON_ARGS

echo "Submitting SiT ReLU Stem..."
submit_job "ft_sit_relu" "ft_sit_stem_relu.out" \
    --model=sit \
    --weights=results/weights/supervised/sit_stem_relu_512/model_30012023_174048 \
    --params=models/configuration/sit_stem_relu.json \
    --use-lora --lora-rank=16 --lora-alpha=32.0 \
    --backbone-lr=1e-5 --middle-lr=5e-5 --head-lr=1e-4 $COMMON_ARGS

echo "All jobs submitted successfully."
