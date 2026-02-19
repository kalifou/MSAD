source activate base
conda activate MSAD

mission="m2"
window_size="256"
save_dir="results/post_finetuning/"
data_dir="data/sample_labelled_esa_adb/"$mission
mode="__TEST_MODE"
model_weights_dir="results/finetuned/terrabyte/"

mkdir -p $save_dir/$mission/$window_size

#srun --nodes=1 --cluster=hpda2 --partition=hpda2_compute --time=03:00:00  --mem=99000 --cpus-per-task=48 \
python3 eval_deep_model.py --model inception_time \
--model_path $model_weights_dir/inception_time_256_20260218_073240_best.pth \
--params models/configuration/inception_time_m2_train.json \
--data $data_dir$"/train_all_targets"$mode$"/ESA_ADB_"$window_size/ \
--path_save $save_dir/$mission/$window_size

#srun --nodes=1 --cluster=hpda2 --partition=hpda2_compute --time=03:00:00  --mem=99000 --cpus-per-task=48 \
python3 eval_deep_model.py --model convnet \
--model_path $model_weights_dir/convnet_256_20260218_091109_best.pth \
--params models/configuration/convnet_m2_train.json \
--data $data_dir$"/train_all_targets"$mode$"/ESA_ADB_"$window_size/ \
--path_save $save_dir/$mission/$window_size