mkdir -p /tmp/evals
mkdir logs_finetuning/

for window_size in 256; do # 128 256 512 1024; do

    model_name=$(basename results/weights/supervised/resnet_default_$window_size/model_*)

    python3 eval_deep_model.py --model=resnet --model_path=results/weights/supervised/resnet_default_$window_size/$model_name \
	    --params=models/configuration/resnet_default.json \
	    --path_save /tmp/di38jul/evals/  \
	    --data data/ESA-ADB/binarized/windowed/m2/train_all_targets__TEST_MODE/ESA_ADB_$window_size/

    # Finetuning SIT
    path_to_sit_weigths=$(basename results/weights/supervised/sit_conv_patch_$window_size/model_*)
    python3 finetune_deep_model.py --path data/ESA-ADB/binarized/windowed/m2/train_all_targets__TEST_MODE/ESA_ADB_$window_size/ \
	    --model=sit \
	    --weights=results/weights/supervised/sit_conv_patch_$window_size/$path_to_sit_weigths \
	    --params=models/configuration/sit_conv_patch.json --use-lora --lora-rank=16 \
	    --lora-alpha=32.0 --layer-wise-lr --backbone-lr=1e-5 --middle-lr=5e-5 --head-lr=1e-4 \
	    --epochs=1 --batch=32 --window-size=$window_size #>> logs_finetuning/logs_sit_$window_size.out 2>&1
    
    best_finetuned_weights_sit=$(grep results/finetuned/weights/sit_$window_size_........_......_best.pth \
				      logs_finetuning/logs_sit_$window_size.out)

    python3 eval_deep_model.py --model=sit --model_path=results/weights/supervised/resnet_default_$window_size/$model_name \
            --params=models/configuration/sit_conv_patch.json \
            --path_save /tmp/di38jul/evals/  \
            --data data/ESA-ADB/binarized/windowed/m2/train_all_targets__TEST_MODE/ESA_ADB_$window_size/
    echo "Post evaluating MSAD baseline on window size: "$window_size
done
