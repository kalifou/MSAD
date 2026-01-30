
source_binarized_data="../anomaly_detection_galileo/preprocessed/multivariate/"
destination_dataset_dir_msad="data/ESA-ADB/binarized/multivariate/"

#"/tmp/di38jul/test_transfer/"
#"data/ESA-ADB/binarized/multivariate/"

ms1_file_prefix="84_months."
ms2_file_prefix="21_months."

# iterate over datasets (mission 1 or 2)
# test vs. train split
# adjust channels of interest
# create directory of interest

for current_dataset in "ESA-Mission1-semi-supervised"; do # "ESA-Mission1-semi-supervised"; do
    if [[ $current_dataset = "ESA-Mission1-semi-supervised" ]]
    then
      echo "Processing dataset 1."
      current_prefix=$ms1_file_prefix
      list_channels=$(seq -s, 12 14) #$(seq -s, 12 52)$','$(seq -s, 57 66)$','$(seq -s, 70 76)
    elif [[ $current_dataset = "ESA-Mission2-semi-supervised" ]]
    then
      echo "Processing Dataset 2."
      current_prefix=$ms2_file_prefix
      list_channels=$(seq -s, 9 11) #$(seq -s, 9 28)$','$(seq -s, 58 59)$','$(seq -s, 70 91)$','$(seq -s, 97 99)
    fi
    
    for split in "train"; do #"test"; do
        if [[ $split = "train" ]]
        then
          echo "Processing train split."
          current_suffix=$split          
        elif [[ $split = "test" ]]
        then
          echo "Processing the test split."
          current_suffix=$split
        fi

        rm -rf $destination_dataset_dir_msad
        echo "deleted content in destination directory..."
        mkdir -p $destination_dataset_dir_msad$current_dataset
        cp $source_binarized_data$current_dataset$"/"$current_prefix$current_suffix$".csv" $destination_dataset_dir_msad$current_dataset$"/"
        echo "Synbolic copy of the data performed into: "$destination_dataset_dir_msad$current_dataset
        echo "Next operating the dataset windowing for these channels..."$list_channels

        for window_size in 128; do # 256 512 1028; do
            echo "Current window size..."$window_size
            python3 create_windows_dataset.py --path $destination_dataset_dir_msad --save_dir $"/tmp/di38jul/windowed/" --metric_path $"data/benchmark_esa_binarized/m1/"$split$"/metrics_working_copy/" --metric R_AUC_PR -w $window_size --name ESA_ADB -chl $list_channels
            echo "done!"
        done        
    done
done