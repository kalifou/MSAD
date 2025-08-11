
srun --nodes=1 --cluster=hpda2 --partition=hpda2_compute --time=00:05:00  --mem=99000 --cpus-per-task=48 \
     python merge_scores.py --path=data/benchmark_esa_binarized/raw_predictions/ --metric=R_AUC_PR \
     --save_path=data/benchmark_esa_binarized/final_results/ --dataset ESA_ADB

srun --nodes=1 --cluster=hpda2 --partition=hpda2_compute --time=00:05:00  --mem=99000 --cpus-per-task=48 \
     python merge_scores.py --path=data/benchmark_esa_binarized/raw_predictions/ --metric=R_AUC_ROC \
     --save_path=data/benchmark_esa_binarized/final_results/ --dataset ESA_ADB
