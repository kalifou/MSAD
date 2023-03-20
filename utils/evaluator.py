########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: utils
# @file : evaluator
#
########################################################################

import os
import pickle
from pathlib import Path
from collections import Counter
from time import perf_counter
from tqdm import tqdm
from datetime import datetime

from utils.timeseries_dataset import TimeseriesDataset
from utils.config import *

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd


class Evaluator:
	"""A class with evaluation tools
	"""

	def predict(
		self,
		model,
		fnames,
		data_path,
		batch_size
	):
		"""Predict function for all the deep learning models
		(ConvNet, ResNet, InceptionTime, SignalTransformer).

		:param model: the object model whose predictions we want
		:param fnames: the names of the timeseries to be predicted
		:param data_path: the path to the timeseries 
			(please check that path and fnames together make the complete path)
		:param batch_size: the batch size used to make the predictions
		:return df: dataframe with timeseries and predictions per time series
		"""

		# Setup
		all_preds = []

		loop = tqdm(
			fnames, 
			total=len(fnames),
			desc=f"Computing({batch_size})",
			unit="files",
			leave=True
		)

		# Main loop
		for fname in loop:
			# Fetch data for this specific timeserie
			data = TimeseriesDataset(
				data_path=data_path,
				fnames=[fname],
				verbose=False
			)
			preds = self.predict_timeseries(model, data, batch_size=batch_size, device='cuda')
			
			# Compute metric value
			counter = Counter(preds)
			most_voted = counter.most_common(1)
			
			# Save info
			all_preds.append(detector_names[most_voted[0][0]])
		
		fnames = [x[:-4] for x in fnames]

		return pd.DataFrame(data=all_preds, columns=["class"], index=fnames)


	def predict_timeseries(self, model, val_data, batch_size, device='cuda', k=1):
		all_preds = []
		
		# Timeserie to batches
		val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

		for (inputs, labels) in val_loader:
			# Move data to the same device as model
			inputs = inputs.to(device)
			labels = labels.to(device)
			
			# Make predictions
			outputs = model(inputs.float())

			# Compute topk acc
			preds = outputs.argmax(dim=1)
			all_preds.extend(preds.tolist())

		return all_preds

def save_classifier(model, path, fname=None):
	# Set up
	timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
	fname = f"model_{timestamp}" if fname is None else fname

	# Create saving dir if we need to
	filename = Path(os.path.join(path, fname))
	filename.parent.mkdir(parents=True, exist_ok=True)

	# Save
	with open(f'{filename}.pkl', 'wb') as output:
		pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


def load_classifier(path):
	"""Loads a classifier/model that is a pickle (.pkl) object.
	If the path is only the path to the directory of a given class
	of models, then the youngest model of that class is retrieved.

	:param path: path to the specific classifier to load,
		or path to a class of classifiers (e.g. rocket)
	:return output: the loaded classifier
	"""

	# If model is not given, load the latest
	if os.path.isdir(path):
		models = [x for x in os.listdir(path) if '.pkl' in x]
		models.sort(key=lambda date: datetime.strptime(date, 'model_%d%m%Y_%H%M%S.pkl'))
		path = os.path.join(path, models[-1])
	elif '.pkl' not in path:
		raise ValueError(f"Can't load this type of file {path}. Only '.pkl' files please")

	filename = Path(path)
	with open(f'{filename}', 'rb') as input:
		output = pickle.load(input)
	
	return output

'''
	def predict_non_deep(self, model, X_val, y_val):
		all_preds = []
		all_acc = []
		
		# Make predictions
		preds = model.predict(X_val)

		# preds = outputs.argmax(dim=1)
		# acc = (y_val == preds).sum() / y_val.shape[0]

		# all_acc.append(acc)
		all_preds.extend(preds.tolist())

		return all_preds
'''