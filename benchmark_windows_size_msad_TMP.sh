mkdir -p /tmp/evals
mkdir logs_finetuning/

for window_size in 256; do # 128 256 512 1024; do

    best_finetuned_weights_sit=$(grep results/finetuned/weights/sit_$window_size_........_......_best.pth \
				      logs_finetuning/logs_sit_$window_size.out)

    echo "Post evaluating MSAD baseline of following weights: "$best_finetuned_weights_sit 
    python3 eval_deep_model.py --model=sit --model_path=results/finetuned/weights/sit_256_20260211_164747_best.pth \
	    --params=models/configuration/sit_conv_patch.json \
            --path_save /tmp/di38jul/evals/  \
            --data data/ESA-ADB/binarized/windowed/m2/test_all_targets/ESA_ADB_$window_size/
    echo "Post evaluating MSAD baseline on window size: "$window_size
done
