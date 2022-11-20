from ift6758.baseline.model_utilities import *
import random
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms


def baseline_models(df_train: pd.DataFrame) :
    '''
    Baseline models (logistic regression)
    
    Args : 
        - df_train : training set

    Return :
        Data for train and validation and dictionary of all predictions and probabilities for each model
    '''

    # Filtering and balancing dataframe
    df_filtered = prepare_df(df_train, ["isGoal", "distanceToGoal", "angleWithGoal"])


    model_list = [["distanceToGoal"], ["angleWithGoal"], ["distanceToGoal", "angleWithGoal"]]
    models = {}

    for i in model_list :
        # Define data for train and validation
        x, y, x_val, y_val = get_train_validation(df_filtered, i, ["isGoal"], 0.2)
        
        
        # Instanciate a logistic regression model
        clf = LogisticRegression()

        # train model
        clf.fit(x, y.reshape(len(y)))
        
        #score model (training set)
        score_training = clf.score(x, y)
        print("Training score for LogisticRegression based on ", " and ".join(i), ": ", score_training)
        
        # score model (validation set)
        score_validation = clf.score(x_val, y_val)
        print("Validation score for LogisticRegression based on ", " and ".join(i), ": ", score_validation)
        print("---------------------------------------------------------------")
        
        # Class predictions and probabilities 
        val_preds = clf.predict(x_val)
        score_prob = clf.predict_proba(x_val)[:, 1]
        models[" and ".join(i)] = {"val_preds" : val_preds, "score_prob" : score_prob}
    return x, y, x_val, y_val, models



def baseline_roc_auc(y_val : np.array, models : dict, save: bool = True, plot: bool = True, path_to_save: str = "./") :
    '''
    Baseline model (logistic regression)
    plot ROC/AUC 
    
    Args :
        - y_val: Labels for validation
        - models: Dictionary of all predictions and probabilities for each model
        - save: Boolean to choose to save the figure
        - plot: choose to plot the figure or not
        - path_to_save : path where to save the figure
    '''

    fig = plt.figure(figsize=[14, 8])
    for i in models :
        #ROC curve and AUC 
        fpr, tpr, threshold = metrics.roc_curve(y_val, models[i]["score_prob"])
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label= i + ' : AUC = %0.2f' % roc_auc)

    score_prob = []
    for i in range(len(y_val)):
        score_prob.append(random.uniform(0, 1))
    score_prob = np.array(score_prob)
    #ROC curve and AUC 
    fpr, tpr, threshold = metrics.roc_curve(y_val, score_prob)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label= "Random Uniform" + ' : AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"ROC_AUC_baseline.png")
    if save or plot:
        plt.close()


def baseline_goal_rate(y_val : np.array, models : dict, save: bool = True, plot: bool = True, path_to_save: str = "./") :
    '''
    Baseline model (logistic regression) based on distance
    plot goal rate
    
    Args : 
        - y_val: Labels for validation
        - models: Dictionary of all predictions and probabilities for each model
        - features: List of columns names of features we want to keep
        - save : Boolean to choose to save the figure
        - plot : choose to plot the figure or not
        - path_to_save : path where to save the figure
    '''
    
    fig, ax = plt.subplots(figsize=[14, 8])
    for i in models :
        rate_array, index_array, goal_array, total_goal = goal_rate(models[i]["val_preds"], models[i]["score_prob"], 5)
        ax.plot(index_array, rate_array, label=i)

    score_prob = []
    for i in range(len(y_val)):
        score_prob.append(random.uniform(0, 1))
    score_prob = np.array(score_prob)
    val_preds = (score_prob >= 0.5).astype('int32')
    ax.plot(index_array, rate_array, label= "Random Uniform")
    plt.xticks(np.arange(0, 110, 10.0))
    plt.yticks(np.arange(0, 110, 10.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.invert_xaxis()
    plt.legend()
    plt.title("Goal Rate")
    plt.xlabel('Shot probability model percentile', fontsize=12)
    plt.ylabel('Goals / (Shots + Goals)', fontsize=12)

    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"goal_rate_baseline.png")
    if save or plot:
        plt.close()


def baseline_goal_rate_cumulative(y_val : np.array, models : dict, save: bool = True, plot: bool = True, path_to_save: str = "./") :
    '''
    Baseline model (logistic regression) based on distance
    plot goal rate cumulative
    
    Args : 
        - y_val: Labels for validation
        - models: Dictionary of all predictions and probabilities for each model
        - features: List of columns names of features we want to keep
        - save : Boolean to choose to save the figure
        - plot : choose to plot the figure or not
        - path_to_save : path where to save the figure
    '''
    
    fig, ax = plt.subplots(figsize=[14, 8])
    for i in models :
        rate_array, index_array, goal_array, total_goal = goal_rate(models[i]["val_preds"], models[i]["score_prob"], 5)
        cumulative_array = compute_cumulative(goal_array, total_goal)
        ax.plot(np.flip(index_array), cumulative_array, label=i)

    score_prob = []
    for i in range(len(y_val)):
        score_prob.append(random.uniform(0, 1))
    score_prob = np.array(score_prob)
    val_preds = (score_prob >= 0.5).astype('int32')
    ax.plot(np.flip(index_array), cumulative_array, label="Random Uniform")
    plt.xticks(np.arange(0, 110, 10.0))
    plt.yticks(np.arange(0, 110, 10.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.invert_xaxis()
    plt.legend()
    plt.title("Cumulative % of goal")
    plt.xlabel('Shot probability model percentile', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)

    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"goal_rate_cumul_baseline.png")
    if save or plot:
        plt.close()


def baseline_calibration(y_val : np.array, models : dict, save: bool = True, plot: bool = True, path_to_save: str = "./") :
    '''
    Baseline model (logistic regression) based on distance
    plot calibration
    
    Args : 
        - y_val: Labels for validation
        - models: Dictionary of all predictions and probabilities for each model
        - features: List of columns names of features we want to keep
        - save : Boolean to choose to save the figure
        - plot : choose to plot the figure or not
        - path_to_save : path where to save the figure
    '''
    
    fig, ax = plt.subplots(figsize=[14, 8])
    for i in models :
        prob_true, prob_pred = calibration_curve(y_val, models[i]["score_prob"], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=i)

    score_prob = []
    for i in range(len(y_val)):
        score_prob.append(random.uniform(0, 1))
    score_prob = np.array(score_prob)
    prob_true, prob_pred = calibration_curve(y_val, score_prob, n_bins=10)

    plt.plot(prob_pred, prob_true, marker='o', label="Random Uniform")
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title('Calibration')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('True probability in each bin')
    plt.legend()

    if plot:
        plt.show()
    if save:
        fig.savefig(path_to_save + f"calibration_baseline.png")
    if save or plot:
        plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    

