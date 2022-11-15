
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.preprocessing import StandardScaler


def baseline_distance(df_train: pd.DataFrame, save: bool = True, plot: bool = True,
                             path_to_save: str = "./") :
    
    '''
    Baseline model (logistic regression) based on distance
    plot ROC/AUC 
    plot calibration curve
    plot calibration curve from predictions and estimator 
    
    Args : 
        - df_train : training set
        - plot : Boolean to choose to plot or not
        - save : Boolean to choose to save the figures
        - path_to_save : Path where the figures will be saved
        
    Returns : 
        - Accuracy of the model 
    '''
    
    #Filter training set on features of interest  
    df_filtered = df_train[['isGoal', 'distanceToGoal']]
    
    #Split to training and validation set
    train, test = train_test_split(df_filtered, test_size=0.33, random_state=42)
    
    #DataFrame to numpy 
    x = train['distanceToGoal'].to_numpy().reshape(-1, 1)
    y = train['isGoal'].to_numpy()
    x_test = test['distanceToGoal'].to_numpy().reshape(-1, 1)
    y_test = test['isGoal'].to_numpy()
    
    
    # instanciate a logistic regression model
    clf = LogisticRegression()

    # train model
    clf.fit(x, y)
    
    #score model (training set)
    score1_training = clf.score(x, y)
    
    # score model (validation set)
    score1_validation = clf.score(x_test, y_test)
    
    #predict class probabilities 
    y_score = clf.predict_proba(x_test)
    
    #ROC curve and AUC 
    preds = y_score[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic based on Distance')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    #Plot and save figure 
    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"Receiver_Operating_Characteristic_Distance.png")
    if save or plot:
        plt.close()
    
    #Calibration curve 
    prob_true, prob_pred = calibration_curve(y_test, preds, n_bins=10)
    disp = CalibrationDisplay(prob_true, prob_pred, preds)
    disp.plot()
    
    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"calibration_curve_Distance.png")
    if save or plot:
        plt.close()
    
    #Calibration curve from predictions 
    disp = CalibrationDisplay.from_predictions(y_test, preds)
    
    #Plot and save figure 
    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"calibration_curve_from_predictions_Distance.png")
    if save or plot:
        plt.close()
    
    #Calibration curve from estimator 
    disp = CalibrationDisplay.from_estimator(clf, x_test, y_test)
    
    #Plot and save figure 
    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"calibration_curve_from_estimator_Distance.png")
    if save or plot:
        plt.close()
        
    return score1_validation
    

def baseline_angle(df_train : pd.DataFrame, save: bool = True, plot: bool = True,
                             path_to_save: str = "./") : 
    
    '''
    Baseline model (logistic regression) based on angle
    plot ROC/AUC 
    plot calibration curve
    plot calibration curve from predictions and estimator 
    
    Args : 
        - df_train : training set
        - plot : Boolean to choose to plot or not
        - save : Boolean to choose to save the figures
        - path_to_save : Path where the figures will be saved
        
    Returns : 
        - Accuracy of the model 
    '''
    
    #Filter training set on features of interest 
    df_filtered_2 = df_train[['isGoal', 'angleWithGoal']]
    
    #Split to training and validation set
    train, test = train_test_split(df_filtered_2, test_size=0.33, random_state=42)
    
    #DataFrame to numpy
    x1 = train[['angleWithGoal']].to_numpy().reshape(-1, 1)
    y1 = train['isGoal'].to_numpy()
    x1_test = test[['angleWithGoal']].to_numpy().reshape(-1, 1)
    y1_test = test['isGoal'].to_numpy()
    
    
    #instanciate a logistic regression model
    clf = LogisticRegression()

    #train model
    clf.fit(x1, y1)
    
    #score model (training set)
    score2_training = clf.score(x1, y1)
    
    #score model (validation set)
    score2_validation = clf.score(x1_test, y1_test)
    
    #Predict class probabilities 
    y_score = clf.predict_proba(x1_test)
    
    #ROC curve and AUC 
    preds = y_score[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y1_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic based on Angle')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    #plot and save the figure 
    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"Receiver_Operating_Characteristic_Angle.png")
    if save or plot:
        plt.close()
    
    #Calibration curve 
    prob_true, prob_pred = calibration_curve(y1_test, preds, n_bins=10)
    disp = CalibrationDisplay(prob_true, prob_pred, preds)
    disp.plot()
    
    #Plot and save figure 
    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"calibration_curve_Angle.png")
    if save or plot:
        plt.close()
    
    #Calibration curve from predictions 
    disp = CalibrationDisplay.from_predictions(y1_test, preds)
    
    #Plot and save figure 
    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"calibration_curve_from_predictions_Angle.png")
    if save or plot:
        plt.close()
    
    #Calibration curve from estimator 
    disp = CalibrationDisplay.from_estimator(clf, x1_test, y1_test)
    
    #Plot and save figure 
    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"calibration_curve_from_estimator_Angle.png")
    if save or plot:
        plt.close()
    
    return score2_validation

    
def baseline_distance_angle(df: pd.DataFrame, save: bool = True, plot: bool = True,
                             path_to_save: str = "./") :
    
    '''
    Baseline model (logistic regression) based on angle and distance
    plot ROC/AUC 
    plot calibration curve
    plot calibration curve from predictions and estimator 
    
    Args : 
        - df_train : training set
        - plot : Boolean to choose to plot or not
        - save : Boolean to choose to save the figures
        - path_to_save : Path where the figures will be saved
        
    Returns : 
        - Accuracy of the model 
    '''
    
    #Filter training set on features of interest 
    df_filtered = df_train[['isGoal', 'distanceToGoal', 'angleWithGoal']]
    
    #Split to training and validation set 
    train, test = train_test_split(df_filtered_3, test_size=0.33, random_state=42)
    
    #DataFrame to numpy 
    x2 = train[['distanceToGoal', 'angleWithGoal']].to_numpy().reshape(-1, 2)
    y2 = train['isGoal'].to_numpy()
    x2_test = test[['distanceToGoal', 'angleWithGoal']].to_numpy().reshape(-1, 2)
    y2_test = test['isGoal'].to_numpy()
    
    #Training and validation set standardization 
    scaler = StandardScaler()
    Z_train = scaler.fit_transform(x2)
    Z_test = scaler.fit_transform(x2_test)
    
    
    #instanciate a logistic regression model
    clf = LogisticRegression()

    #train model
    clf.fit(Z_train, y2)
    
    #score model (training set)
    score3_training = clf.score(Z_train, y2)
    
    #score model (validation set)
    score3_validation = clf.score(Z_test, y2_test)
    
    #Predit class probabilities 
    y_score = clf.predict_proba(Z_test)
    
    #ROC curve and AUC 
    preds = y_score[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y2_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic based on Angle/Distance')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    #Plot and save figure 
    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"Receiver_Operating_Characteristic_Angle_Distance.png")
    if save or plot:
        plt.close()
    
    #Calibration curve 
    prob_true, prob_pred = calibration_curve(y2_test, preds, n_bins=10)
    disp = CalibrationDisplay(prob_true, prob_pred, preds)
    disp.plot()
    
    #Plot and save figure 
    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"calibration_curve_Angle_Distance.png")
    if save or plot:
        plt.close()
    
    #Calibration cruve from predictions 
    disp = CalibrationDisplay.from_predictions(y2_test, preds)
    
    #Plot and save figure 
    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"calibration_curve_from_predictions_Angle_Distance.png")
    if save or plot:
        plt.close()
    
    #Calibration cruve from estimator 
    disp = CalibrationDisplay.from_estimator(clf, Z_test, y2_test)
    
    #Plot and save figure
    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"calibration_curve_from_estimator_Angle_Distance.png")
    if save or plot:
        plt.close()
    
    return score3_validation 
    
    
    
    
    
    
    
    
    
    
    
    
    

