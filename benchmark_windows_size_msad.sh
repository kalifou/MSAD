mkdir -p /tmp/evals
mkdir logs_finetuning/

path_dataset_local="data/ESA-ADB/binarized/windowed/m2/" #"data/sample_labelled_esa_adb/m2/" #"data/ESA-ADB/binarized/windowed/m2/"
save_path="/tmp/di38jul/evals/"
num_epochs=1
batch_size=32
mode="" #__TEST_MODE

mkdir -p $save_path

for window_size in 256; do # 128 256 512 1024; do

    model_name=$(basename results/weights/supervised/resnet_default_$window_size/model_*)

    #python3 eval_deep_model.py --model=resnet --model_path=results/weights/supervised/resnet_default_$window_size/$model_name \
	#    --params=models/configuration/resnet_default.json \
	#    --path_save $save_path  \
	#    --data $path_dataset_local/train_all_targets$mode/ESA_ADB_$window_size/

    # Finetuning SIT
    
    model_stem="inception_time" 
    rm $"logs_finetuning/logs_"$model_stem"_"$window_size.out
    path_to_model_weigths=$(basename results/weights/supervised/$model_stem$"_default_"$window_size/model_*)
    echo "Loading the following weights for: "$path_to_model_weigths
    python3 finetune_deep_model.py --path $path_dataset_local/train_all_targets/ESA_ADB_$window_size/ \
	    --model=$model_stem \
	    --weights=results/weights/supervised/$model_stem"_default_"$window_size/$path_to_model_weigths \
	    --params=models/configuration/$model_stem$"_default.json"  --use-lora --lora-rank=16 \
	    --lora-alpha=32.0 --layer-wise-lr --backbone-lr=1e-5 --middle-lr=5e-5 --head-lr=1e-4 \
	    --epochs=$num_epochs --batch=$batch_size --window-size=$window_size #>> logs_finetuning/logs_$model_stem$"_"$window_size.out 2>&1
    
    #sit: sit_conv_patch.json
    
    best_finetuned_weights_current=$(grep -Po results/finetuned/weights/$model_stem$"_"$window_size........._......_best.pth \
				      logs_finetuning/logs_$model_stem"_"$window_size.out)
    echo "Generated the following weights for our model: "$best_finetuned_weights_current
    
    python3 eval_deep_model.py --model=$model_stem --model_path=$best_finetuned_weights_current \
            --params=models/configuration/$model_stem$"_m2_train.json"  \
            --path_save $save_path  \
            --data $path_dataset_local/train_all_targets/ESA_ADB_$window_size/
    echo "Post evaluating MSAD baseline on window size: "$window_size
done
