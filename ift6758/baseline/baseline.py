from ift6758.utilities.model_utilities import *
from sklearn.linear_model import LogisticRegression
from comet_ml import Experiment
import pickle
import os


def baseline_models(df_train: pd.DataFrame, project_name: str, workspace: str, comet : bool = True, balanced : bool = True) :
    '''
    Baseline models (logistic regression)
    
    Args : 
        df_train : training set
        project_name: Name of project
        workspace: Name of workspace
        commet: Bool to decide to register model on comet or not
        balanced: Booléan to tell the function to balanced data or not

    Return :
        Data for train and validation, dictionary of all predictions and probabilities for each model and all commet experiments
    '''

    # Filtering and balancing dataframe
    df_filtered = prepare_df(df_train, ["isGoal", "distanceToGoal", "angleWithGoal"])


    model_list = [["distanceToGoal"], ["angleWithGoal"], ["distanceToGoal", "angleWithGoal"]]
    models = {}
    experiments = {}

    for i in model_list :
        if comet :
            experiment = Experiment(
                api_key=os.environ.get('COMET_API_KEY'),
                project_name=project_name,
                workspace=workspace
            )
            experiment.set_name("LogisticRegression_" + "_".join(i))
            experiments["LogisticRegression_" + "_".join(i)] = experiment
        # Define data for train and validation
        x, y, x_val, y_val = get_train_validation(df_filtered, i, ["isGoal"], 0.2, balanced)
        
        
        # Instanciate a logistic regression model
        clf = LogisticRegression(random_state=42)

        # train model
        clf.fit(x, y)
        pickle.dump(clf, open("ift6758/models/LogisticRegression_" + "_".join(i) + ".pkl", "wb"))
        
        #score model (training set)
        score_training = clf.score(x, y)
        
        # score model (validation set)
        score_validation = clf.score(x_val, y_val)
        
        # Class predictions and probabilities 
        val_preds = clf.predict(x_val)
        score_prob = clf.predict_proba(x_val)[:, 1]
        f1 = f1_score(y_val, val_preds, average="macro")
        models["LogisticRegression_" + "_".join(i)] = {"val_preds" : val_preds, "score_prob" : score_prob, "f1" : f1}

        if comet :
            experiment.log_model("LogisticRegression_" + "_".join(i), "ift6758/models/LogisticRegression_" + "_".join(i) + ".pkl")
            experiment.log_metric("train_score", score_training)
            experiment.log_metric("validation_score", score_validation)
            experiment.log_metric("f1_score", f1)
            experiment.log_confusion_matrix(y_val.astype('int32'), val_preds.astype('int32'))
            experiment.end()
    return (x, y, x_val, y_val), models, experiments
    
    
    
    
    
    
    
    
    
    
    
    
    

