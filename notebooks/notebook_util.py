import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve, CalibrationDisplay

import matplotlib.ticker as mtick
import seaborn as sns

from imblearn.over_sampling import RandomOverSampler


feature = ['speedOfChangeOfAngle', 'speed', 'changeOfAngleFromPrev', 'rebound',
       'distanceFromPrev', 'secondsSincePrev', 'prevAngleWithGoal', 'prevY',
       'prevX', 'prevSecondsSinceStart', 'angleWithGoal', 'distanceToGoal',
       'x', 'y', 'emptyNet', 'secondsSinceStart', 'strength_Even',
       'strength_Power Play', 'strength_Short Handed', 'shotType_Backhand',
       'shotType_Deflected', 'shotType_Slap Shot', 'shotType_Snap Shot',
       'shotType_Tip-In', 'shotType_Wrap-around', 'shotType_Wrist Shot',
       'prevEvent_Blocked Shot', 'prevEvent_Faceoff', 'prevEvent_Giveaway',
       'prevEvent_Goal', 'prevEvent_Hit', 'prevEvent_Missed Shot',
       'prevEvent_Penalty', 'prevEvent_Shot', 'prevEvent_Takeaway']

    

def prep_data():
    df = pd.read_csv('../ift6758/data/extracted/shot_goal_20151007_20210707.csv')

    df_dropped = df[(df['season'].isin([20152016, 20162017, 20172018, 20182019])) & (df['gameType'] == 'R') & (
                df['periodType'] != 'SHOOTOUT')]

    df_filtered = df_dropped[['speedOfChangeOfAngle', 'speed', 'changeOfAngleFromPrev', 'rebound', 'distanceFromPrev'
                            , 'secondsSincePrev', 'prevAngleWithGoal', 'prevY', 'prevX', 'prevEvent', 'prevSecondsSinceStart',
                            'angleWithGoal', 'distanceToGoal', 'x', 'y', 'emptyNet', 'strength', 'secondsSinceStart', 'shotType', 'isGoal']]

    columns_count = len(df_filtered.columns) - 1

    df_filtered['emptyNet'] = df_filtered['emptyNet'].fillna(0)
    df_filtered['strength'] = df_filtered['strength'].fillna('Even')
    df_filtered = df_filtered.dropna()

    df_filtered.head(1)

    return df_filtered

def prep_dummie(df_filtered):
    dummy_object = pd.get_dummies(df_filtered[['strength', 'shotType', 'prevEvent']])
    df_filtered = df_filtered.merge(dummy_object, left_index=True, right_index=True)
    df_filtered = df_filtered.drop(labels = ['strength', 'shotType', 'prevEvent'], axis = 1)

    return df_filtered

def show_roc(fpr, tpr, roc_auc):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1])
    plt.ylim([0, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def prepare_goal_rate(y_array, score_prob):
    built_array = []
    index_array = list(range(0, 100, 5))
    total_goal = 0
    goal_array = []
    for i in range(0, 100, 5) :
        sub_y_array = y_array[np.logical_and(score_prob >= np.percentile(score_prob, i), score_prob < np.percentile(score_prob, i+5))]
        goals = np.count_nonzero(sub_y_array)
        shots = sub_y_array.size - goals
        sub_final = goals / (shots + goals)
        goal_array.append(goals)
        total_goal = total_goal + goals
        built_array.append(sub_final*100)
    return built_array, index_array, goal_array, total_goal

def create_goal_rate_plot(index_array, built_array):
    fig, ax = plt.subplots()
    ax.plot(index_array, built_array, color="blue")
    plt.xticks(np.arange(0, 110, 10.0))
    plt.yticks(np.arange(0, 110, 10.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.invert_xaxis()
    ax.legend(['Baseline'])
    plt.title("Goal Rate")
    plt.xlabel('Shot probability model percentile', fontsize=12)
    plt.ylabel('Goals / (Shots + Goals)', fontsize=12)
    plt.show()

def compute_cumulative(goal_array, total_goal):
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

def create_cumulative_plot(index_array, cumulative_array):
    fig, ax = plt.subplots()
    ax.plot(index_array, cumulative_array, color="blue")
    plt.xticks(np.arange(0, 110, 10.0))
    plt.yticks(np.arange(0, 110, 10.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.invert_xaxis()
    ax.legend(['Baseline'])
    plt.title("Cumulative % of goal")
    plt.xlabel('Shot probability model percentile', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.show()