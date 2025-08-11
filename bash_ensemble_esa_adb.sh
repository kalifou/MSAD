
#3.1 AVG ENS
for channel in 33 32 40 46 44 45 61 62 63; do
    srun --nodes=1 --cluster=hpda2 --partition=hpda2_compute --time=00:30:00  --mem=99000 --cpus-per-task=48 \
	 nohup python run_avg_ens__esa_adb.py -d ESA_ADB -ch $channel >> avg_$channel.out &
done
#3.2 Oracles - DONE V
srun --nodes=1 --cluster=hpda2 --partition=hpda2_compute_gpu --gres=gpu:1 --time=00:30:00  --mem=99000 --cpus-per-task=48 \
     nohup python run_oracle.py --path=data/benchmark_esa_binarized/metrics/ --acc=1.0 --randomness=true >> oracle.out &

#4 Prepare the Model Selection
#4.1 windowing the dataset

srun --nodes=1 --cluster=hpda2 --partition=hpda2_compute --time=03:00:00  --mem=99000 --cpus-per-task=48 \
     nohup python create_windows_dataset.py --name ESA_ADB --save_dir=data/ESA-ADB/binarized/multivariate/windowed/ \
     --path=data/ESA-ADB/binarized/multivariate/ --metric_path=data/benchmark_esa_binarized/metrics/ --window_size=512 \
     --metric=R_AUC_PR --channel_list 33,32,40,46,44,45,61,62,63 >> windowed.out &
