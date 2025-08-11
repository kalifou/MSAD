########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : create_windows_dataset
#
########################################################################


import sys
import os
from tqdm import tqdm
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from utils.data_loader import DataLoader
from utils.metrics_loader import MetricsLoader
from utils.scores_loader import ScoresLoader
from utils.config import *

import ipdb

def create_tmp_dataset(
    name,
    save_dir,
    data_path,
    metric_path,
    metric, 
    channel_list
):
    """Generates a new dataset from the given dataset. The time series
    in the generated dataset have been divided in windows.

    :param name: the name of the experiment
    :param save_dir: directory in which to save the new dataset
    :param data_path: path to dataset to be divided
    :param window_size: the size of the window timeseries will be split to
    :param metric: the specific metric to read
    """

    # Load datasets
    dataloader = DataLoader(data_path)
    datasets = dataloader.get_dataset_names()
    
    if "TSB_UAD" in name:
        x, y, fnames = dataloader.load(datasets)
    else:
        #ipdb.set_trace(context=25)
        x, y, fnames = dataloader.load_esa_adb_multichannel_df(datasets, 
                                                               channel_list, 
                                                               test_mode=True)

    # Load metrics
    metricsloader = MetricsLoader(metric_path)
    metrics_data = metricsloader.read(metric)
    #ipdb.set_trace(context=55)

    # Delete any data not in metrics (some timeseries metric scores were not computed)
    idx_to_delete = [i for i, x in enumerate(fnames) if x not in metrics_data.index]

    #ipdb.set_trace(context=25)
    metrics_data = metrics_data[detector_names]
   
    filename_merged_metrics = "mergedTable_" + metric + ".csv"
    #ipdb.set_trace(context=55)
    
    if "ESA_ADB" in name:
        filename_merged_metrics = filename_merged_metrics.split("_R")
        filename_merged_metrics = filename_merged_metrics[0] + filename_merged_metrics[1]
        
    metrics_data.to_csv(os.path.join(save_dir, filename_merged_metrics))



# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Create temporary/experiment-specific dataset',
        description='This function creates a dataset of the size you want.  The data that will be used are set into the config file',
        epilog='Be careful where you save the generated dataset'
    )

    parser.add_argument('-n', '--name', type=str, help='path to save the dataset', required=True)
    parser.add_argument('-s', '--save_dir', type=str, help='path to save the merge', required=True)
    parser.add_argument('-p', '--path', type=str, help='path of the dataset of reference', required=True)
    parser.add_argument('-mp', '--metric_path', type=str, help='path to the metrics of the dataset given', default=TSB_metrics_path)
    parser.add_argument('-m', '--metric', type=str, help='metric to use to produce the labels', default='AUC_PR')
    parser.add_argument('-chl', '--channel_list', type=list_of_ints, default=[])

    args = parser.parse_args()
    
    assert args.name in  ["TSB_UAD", "ESA_ADB"]
    
    channel_list = args.channel_list
    if args.name == "ESA_ADB":
        assert channel_list !=list()
        for ch_i in channel_list:
            assert ch_i >=1 and ch_i <= 76
    else:
        assert channel_list == list()
    
    
    create_tmp_dataset(
        name=args.name,
        save_dir=args.save_dir,
        data_path=args.path,
        metric_path=args.metric_path,
        metric=args.metric,
        channel_list=channel_list
    )

