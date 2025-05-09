########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : run_avg_ens
#
########################################################################


from models.model.avg_ens import Avg_ens

from utils.scores_loader import ScoresLoader
from utils.data_loader import DataLoader
from utils.metrics_loader import MetricsLoader
from utils.config import TSB_metrics_path, TSB_data_path, TSB_scores_path, \
    ESA_ADB_metrics_path, ESA_ADB_data_path, ESA_ADB_scores_path

import argparse
import numpy as np
import sys


def create_avg_ens(n_jobs=1, source_dataset="TSB_UAD"):
	'''Create, fit and save the results for the 'Avg_ens' model

	:param n_jobs: Threads to use in parallel to compute the metrics faster
	'''
    
	if source_dataset == "TSB_UAD":
		metrics_path = TSB_metrics_path
		data_path = TSB_data_path
		scores_path = TSB_scores_path
	else:
		metrics_path = ESA_ADB_metrics_path
		data_path = ESA_ADB_data_path
		scores_path = ESA_ADB_scores_path
	# Load metrics' names
	metricsloader = MetricsLoader(metrics_path)
	metrics = metricsloader.get_names()

	# Load data
	dataloader = DataLoader(data_path)
	datasets = dataloader.get_dataset_names()
	x, y, fnames = dataloader.load(datasets)

	# Load scores
	scoresloader = ScoresLoader(scores_path)
	scores, idx_failed = scoresloader.load(fnames)

	# Remove failed idxs
	if len(idx_failed) > 0:
		for idx in sorted(idx_failed, reverse=True):
			del x[idx]
			del y[idx]
			del fnames[idx]

	# Create Avg_ens
	avg_ens = Avg_ens()
	metric_values = avg_ens.fit(y, scores, metrics, n_jobs=n_jobs)
	for metric in metrics:
		# Write metric values for avg_ens
		metricsloader.write(metric_values[metric], fnames, 'AVG_ENS', metric)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='run_avg_ense',
		description="Create the average ensemble model"
	)
	parser.add_argument('-n', '--n_jobs', type=int, default=4,
		help='Threads to use for parallel computation'
	)
	parser.add_argument('-d', '--dataset', type=str, help='Data used for the experiments of individual anomaly detectors. Option: <ESA_ADB, TSB_UAD>.', required=True)
    
	args = parser.parse_args()
	n_jobs, source_dataset = args.n_jobs, args.dataset
	assert source_dataset in ["TSB_UAD","ESA_ADB"]
	
	create_avg_ens(n_jobs=n_jobs, source_dataset=source_dataset)