import os
import pickle

import torch
from comet_ml import Experiment
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from torch import nn
from torch.autograd import Variable
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import BinaryF1Score

from nhlanalysis.utilities.model_utilities import *


def mlp_model(df_train: pd.DataFrame, features: list, model_name: str, project_name: str, workspace: str,
              net, optimizer, hyper_params: dict, comet: bool = True, balanced: bool = True):
    '''
    MLP model

    Args :
        df_train: training set
        features: list of features
        model_name: name of the model
        project_name: Name of project
        workspace: Name of workspace
        net: neural network to use
        optimizer: optimizer to use
        hyper_params: hyperparameters to use
        comet: Bool to decide to register model on comet or not
        balanced: Booléan to tell the function to balanced data or not

    Return :
        Data for train and validation, dictionary of all predictions and probabilities for each model and all commet experiments
    '''

    # Filtering and balancing dataframe
    df_filtered = filter_and_one_hot_encode_features(df_train, features, 'R')

    experiment = None
    if comet:
        experiment = Experiment(
            api_key=os.environ.get('COMET_API_KEY'),
            project_name=project_name,
            workspace=workspace
        )
        experiment.set_name(model_name)
    # Define data for train and validation
    features = list(df_filtered.columns)
    f_remove = features
    f_remove.remove("isGoal")
    x_train, y_train, x_test, y_test = get_train_validation(df_filtered, f_remove, ["isGoal"], 0.33, balanced, 'over')

    batch_size = hyper_params["batch_size"]
    num_epochs = hyper_params["num_epochs"]
    learning_rate = hyper_params["learning_rate"]
    batch_no = len(x_train) // batch_size

    criterion = nn.BCELoss()

    x_train_t = torch.tensor(np.vstack(x_train).astype(np.float32), dtype=torch.float32)
    y_train_t = torch.tensor(y_train.astype(np.float32), dtype=torch.float32)

    for epoch in range(num_epochs):
        if epoch % 5 == 0:
            print('Epoch {}'.format(epoch + 1))
        x_train_t, y_train_t = shuffle(x_train_t, y_train_t)
        loss_sum = 0
        for i in range(batch_no):
            start = i * batch_size
            end = start + batch_size
            x_var = Variable(torch.FloatTensor(x_train_t[start:end]))
            y_var = Variable(torch.FloatTensor(y_train_t[start:end]))
            optimizer.zero_grad()
            ypred_var = net(x_var)
            loss = criterion(ypred_var, y_var[:, None])
            loss.backward()
            loss_sum = loss_sum + loss
            # print(loss)
            optimizer.step()
        print('Loss Sum: ', loss_sum / batch_no)

    # Evaluate the model
    accuracy, f1, result, values = evaluate_model(net, x_test, y_test)

    pickle.dump(net, open(f"./models/{model_name}.sav", 'wb'))

    model = {}
    model[model_name] = {"model": net, "values": values, "score_prob": result.numpy(),
                         "f1": f1}

    if comet:
        experiment.log_model(model_name, "./models/" + model_name + ".sav")
        experiment.log_parameters(hyper_params)
        experiment.log_metric("f1", f1)
        experiment.log_metric("accuracy", accuracy)
        experiment.log_parameters(hyper_params)
        experiment.end()

    return (x_train, y_train, x_test, y_test), model, experiment


def evaluate_model(net, x_test, y_test):
    x_test_t = torch.tensor(np.vstack(x_test).astype(np.float32), dtype=torch.float32)
    y_test_t = torch.tensor(y_test.astype(np.float32), dtype=torch.float32)

    test_var = Variable(torch.FloatTensor(x_test_t), requires_grad=True)
    with torch.no_grad():
        result = net(test_var)

    values = torch.round(result[:, 0])
    num_right = np.sum(values.data.numpy().astype(int) == y_test)
    print('Num Right', num_right)
    accuracy = num_right / len(y_test_t)
    print('Accuracy {:.2f}'.format(accuracy))

    target_m = torch.tensor(y_test).to(torch.int)
    pred_m = torch.tensor(values).to(torch.int)

    confmat = ConfusionMatrix(num_classes=2)
    confmat(target_m, pred_m)

    metric = BinaryF1Score()
    print('F1 Pytorch')
    print(metric(result[:, 0], target_m).item())

    f1 = f1_score(target_m.numpy(), values.numpy().astype(int), average='macro')
    print('F1 Sklearn Macro')
    print(f1)

    return accuracy, f1, result[:, 0], values.numpy().astype(int)
