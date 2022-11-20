import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve, CalibrationDisplay

from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.ticker as mtick
import seaborn as sns

sns.set()

def prepare_df(df : pd.DataFrame, features : list, balanced : bool = True) :
	"""
	Prepare data frame to be use. This function permite to only keep columns that we want and to balanced data if necessary.

	Args:
		df: Data frame for the model
		features: List of columns names of features we want to keep
		balanced: Booléan to tell the function to balanced data or not
	Returns:
		Data frame 
	"""
	df_dropped = df[(df['gameType'] == 'R') & (df['periodType'] != 'SHOOTOUT')]
	df_filtered = df_dropped[features]
	if balanced :
		arg_non_goal = df_filtered[df_filtered["isGoal"] == 0].index
		arg_to_delete = np.random.choice(arg_non_goal,
			size=len(df_filtered[df_filtered["isGoal"] == 0]) - len(df_filtered[df_filtered["isGoal"] == 1]),
			replace=False)
		df_filtered = df_filtered.drop(labels=arg_to_delete, axis=0)
	return df_filtered

def get_train_validation(df : pd.DataFrame, data_features : list, labels_features : list, val_ratio : float) :
	"""
	Get train and validation dataset. You can choose the size of each dataset and the column for labels and data.

	Args:
		df: Data frame for the model
		data_features: List of columns to be use for the inputs.
		labels_features: List of columns to be use for labels
		val_ratio: Size of the validation dataset
	Returns:
		x_train, y_train, x_val, y_val
	"""
	train, val = train_test_split(df, test_size=val_ratio, random_state=42)
	
	x = train[data_features].to_numpy().reshape(-1, len(data_features))
	y = train[labels_features].to_numpy()

	x_val = val[data_features].to_numpy().reshape(-1, len(data_features))
	y_val = val[labels_features].to_numpy()
	return x, y, x_val, y_val

def goal_rate(predictions : np.array, score_prob : np.array, bin_size : int) :
	"""
	Get data to make goal rate graphic ans cumulative graphic. You can choose the size of the bins of percentage.

	Args:
		predictions: Predictions from the model
		score_prob: Probabilities of predictions
		bin_size: Size of bins of percentage
	Returns:
		rate_array: array of goal rate
		index_rate: index in percentage for each goal rate
		goal_array: number of goals for each index
		total_array: total of goals and shots for each index
	"""
	rate_array = []
	index_array = list(range(0, 100, bin_size))
	total_goal = 0
	goal_array = []
	for i in range(0, 100, bin_size) :
	    sub_array = predictions[np.logical_and(score_prob >= np.percentile(score_prob, i), score_prob < np.percentile(score_prob, i+bin_size))]
	    goals = np.count_nonzero(sub_array)
	    shots = sub_array.size - goals
	    sub_final = goals / (shots + goals)
	    goal_array.append(goals)
	    total_goal = total_goal + goals
	    rate_array.append(sub_final*100)
	return rate_array, index_array, goal_array, total_goal

def compute_cumulative(goal_array : list, total_goal : list):
	"""
	Get cumulative data for cumulative goal rate graphic.

	Args:
		goal_array: number of goals for each index
		total_array: total of goals and shots for each index
	Returns:
		cumlative_array
	"""
	cumulative_array = []
	last_elem = 0
	for i in np.flip(goal_array):
	    if total_goal != 0:
	        current = i / total_goal*100 + last_elem
	    else:
	        current = last_elem
	    cumulative_array.append(current)
	    
	    last_elem = current
	return cumulative_array
