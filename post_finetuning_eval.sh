source activate base
conda activate MSAD_ESA

save_dir="results/post_finetuning/"
window_size="256"
mode="" # "__TEST_MODE/" 
model_weights_dir="results/finetuned/weights/" #"results/finetuned/terrabyte/"

for mission in "m1"; do # "m1" "m2" ; do 

    data_dir="data/ESA-ADB/binarized/windowed/"$mission #"data/sample_labelled_esa_adb/"$mission
    
    mkdir -p $save_dir/$mission/$window_size
    
    nohup srun --nodes=1 --cluster=hpda2 --partition=hpda2_compute --time=05:00:00  --mem=99000 --cpus-per-task=96 \
    python3 eval_deep_model.py --model inception_time \
    --model_path $model_weights_dir/inception_time_256_20260218_073240_best.pth \
    --params models/configuration/inception_time_m2_train.json \
    --data $data_dir$"/test_all_targets"$mode$"/ESA_ADB_"$window_size/ \
    --path_save $save_dir/$mission/$window_size >> "finetuning_inception_"$mission"_"$window_size"_"$mode".out" &
    
    nohup srun --nodes=1 --cluster=hpda2 --partition=hpda2_compute --time=05:00:00  --mem=99000 --cpus-per-task=96 \
    python3 eval_deep_model.py --model convnet \
    --model_path $model_weights_dir/convnet_256_20260218_091109_best.pth \
    --params models/configuration/convnet_m2_train.json \
    --data $data_dir$"/test_all_targets"$mode$"/ESA_ADB_"$window_size/ \
    --path_save $save_dir/$mission/$window_size >> "finetuning_convnet_"$mission"_"$window_size"_"$mode".out" &

done