import os
from datetime import datetime

import pandas as pd
import torch

from nhlanalysis.custom_models.knn import knn_model
from nhlanalysis.custom_models.mlp import mlp_model, evaluate_model
from nhlanalysis.custom_models.net_adam import NetAdam
from nhlanalysis.custom_models.net_sgd import NetSGD

pd.options.mode.chained_assignment = None  # default='warn'

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from sklearn.feature_selection import f_classif, mutual_info_classif

from nhlanalysis.features.data_extractor import extract_and_cleanup_play_data, add_previous_event_for_shots_and_goals
from nhlanalysis.features.feature_engineering import log_dataframe_profile
from nhlanalysis.features.incorrect_feature_analysis import incorrect_feature_analysis
from nhlanalysis.visualizations.advanced_visualizations import generate_interactive_shot_map, generate_static_shot_map
from nhlanalysis.visualizations.features_engineering import shots_and_goals_by_distance, shots_and_goals_by_angles, \
    shots_by_angles_and_distance, goal_ratio_by_distance, goal_ratio_by_angles, empty_goal_by_distance
from nhlanalysis.visualizations.simple_visualizations import shots_efficiency_by_type, shots_efficiency_by_distance, \
    shots_efficiency_by_type_and_distance
from nhlanalysis.baseline.baseline import baseline_models
from nhlanalysis.xgboost.xgboost import best_hyperparameters, xgboost_model
from nhlanalysis.utilities.model_utilities import roc_auc_curve, goal_rate_curve, goal_rate_cumulative_curve, calibration, \
    download_model_from_comet, filter_and_one_hot_encode_features, split_data_and_labels, load_model_from_file

if __name__ == "__main__":
    # Start of Milestone 1
    start_date = datetime(2015, 10, 7)  # Start of 2015-2016
    end_date = datetime(2021, 7, 7)  # End of 2020-2021

    outdir = 'nhlanalysis/data/extracted'
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    filename = f'all_events_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv'
    path = os.path.join(outdir, filename)
    if not os.path.exists(path):
        # Extract data for all events between start_date and end_date
        # Extracted data also includes information that may be useful for later milestones
        print("Downloading/Reading...")
        all_plays_df = extract_and_cleanup_play_data(start_date, end_date,
                                                     columns_to_keep=['gameId', 'season', 'gameType', 'dateTime',
                                                                      'team', 'eventIdx', 'event', 'isGoal',
                                                                      'secondaryType', 'description', 'period',
                                                                      'periodType', 'periodTime', 'secondsSinceStart',
                                                                      'players', 'strength', 'emptyNet', 'x', 'y',
                                                                      'rinkSide', 'distanceToGoal', 'angleWithGoal'])

        # Save the unfiltered data
        print("Saving all events DataFrame...")
        all_plays_df.to_csv(os.path.join(outdir, filename), index=False)
    else:
        all_plays_df = pd.read_csv(path)

    # Convert all numeric types
    all_plays_df = all_plays_df.apply(pd.to_numeric, args=('ignore',))

    # Filter out for shots and goals only, data frame will later be used for feature engineering
    filename = f'shot_goal_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv'
    path = os.path.join(outdir, filename)
    if not os.path.exists(os.path.join(outdir, filename)):
        all_plays_df_filtered = add_previous_event_for_shots_and_goals(all_plays_df)
        all_plays_df_filtered.dropna(how='all', axis=1, inplace=True)  # Drop columns that are completely null
        # Save the extracted data into a csv file to skip the long execution time to extract data for visualizations
        print("Saving filtered events DataFrame...")
        all_plays_df_filtered.to_csv(os.path.join(outdir, filename), index=False)
    else:
        all_plays_df_filtered = pd.read_csv(path)

    # Convert all numeric types
    all_plays_df_filtered = all_plays_df_filtered.apply(pd.to_numeric, args=('ignore',))

    print("Generating visualizations...")
    # Make and save simple visualizations in ./figures directory
    shots_efficiency_by_type(all_plays_df_filtered, 20182019, plot=False, path_to_save="./figures/")
    shots_efficiency_by_distance(all_plays_df_filtered, [20182019, 20192020, 20202021], plot=False,
                                 path_to_save="./figures/")
    shots_efficiency_by_type_and_distance(all_plays_df_filtered, 20182019, plot=False, path_to_save="./figures/")

    # Make and save advanced visualizations in ./figures directory
    [generate_interactive_shot_map(all_plays_df_filtered, season, plot=False, path_to_save="./figures/") for season in
     all_plays_df_filtered['season'].unique()]

    [generate_static_shot_map(all_plays_df_filtered, 'Colorado Avalanche', season, plot=False,
                              path_to_save="./figures/") for season in [20162017, 20202021]]
    [generate_static_shot_map(all_plays_df_filtered, team, season, plot=False,
                              path_to_save="./figures/") for team in ['Buffalo Sabres', 'Tampa Bay Lightning'] for
     season in [20182019, 20192020, 20202021]]

    # Start of Milestone 2
    # Create training and test data frames
    df_train = all_plays_df_filtered[
        (all_plays_df_filtered['season'].isin([20152016, 20162017, 20172018, 20182019])) & (
                all_plays_df_filtered['gameType'] == 'R') & (
                all_plays_df_filtered['periodType'] != 'SHOOTOUT')]

    df_test_regular = all_plays_df_filtered[
        (all_plays_df_filtered['season'] == 20192020) & (all_plays_df_filtered['gameType'] == 'R') & (
                all_plays_df_filtered['periodType'] != 'SHOOTOUT')]
    df_test_regular["emptyNet"] = df_test_regular["emptyNet"].fillna(0)
    df_test_regular["strength"] = df_test_regular["strength"].fillna('Even')

    df_test_playoffs = all_plays_df_filtered[
        (all_plays_df_filtered['season'] == 20192020) & (all_plays_df_filtered['gameType'] == 'P')]
    df_test_playoffs["emptyNet"] = df_test_playoffs["emptyNet"].fillna(0)
    df_test_playoffs["strength"] = df_test_playoffs["strength"].fillna('Even')

    # Run incorrect feature analysis and generate the relevant CSVs
    incorrect_feature_analysis(df_train)

    print("Generating features engineering visualizations...")
    shots_and_goals_by_distance(df_train, plot=False, path_to_save="./figures/")
    shots_and_goals_by_angles(df_train, plot=False, path_to_save="./figures/")
    shots_by_angles_and_distance(df_train, plot=False, path_to_save="./figures/")
    goal_ratio_by_distance(df_train, plot=False, path_to_save="./figures/")
    goal_ratio_by_angles(df_train, plot=False, path_to_save="./figures/")
    empty_goal_by_distance(df_train, plot=False, path_to_save="./figures/")

    log_dataframe_profile(df_train[df_train['gameId'] == 2017021065], 'feature_engineering_data',
                          'ift6758a-a22-g3-projet', 'wpg_v_wsh_2017021065', 'csv')

    print("Baseline model...")
    (x, y, x_val, y_val), models, experiments = baseline_models(df_train, 'baseline_models', 'ift6758a-a22-g3-projet',
                                                                comet=True)
    roc_auc_curve(y_val, models, plot=False, path_to_save="./figures/", model_name="baseline")
    goal_rate_curve(y_val, models, plot=False, path_to_save="./figures/", model_name="baseline")
    goal_rate_cumulative_curve(y_val, models, plot=False, path_to_save="./figures/", model_name="baseline")
    calibration(y_val, models, plot=False, path_to_save="./figures/", model_name="baseline")

    print("Baseline test...")
    LR_d = models["LogisticRegression_distanceToGoal"]["model"]
    LR_a = models["LogisticRegression_angleWithGoal"]["model"]
    LR_da = models["LogisticRegression_distanceToGoal_angleWithGoal"]["model"]

    test_reg_models = {}
    test_poff_models = {}

    y_test_reg = df_test_regular["isGoal"].to_numpy().reshape(-1)
    y_test_poff = df_test_playoffs["isGoal"].to_numpy().reshape(-1)

    x_test_reg = df_test_regular["distanceToGoal"].to_numpy().reshape(-1, 1)
    x_test_poff = df_test_playoffs["distanceToGoal"].to_numpy().reshape(-1, 1)
    val_preds_reg_d = LR_d.predict(x_test_reg)
    val_preds_poff_d = LR_d.predict(x_test_poff)
    score_prob_reg_d = LR_d.predict_proba(x_test_reg)[:, 1]
    score_prob_poff_d = LR_d.predict_proba(x_test_poff)[:, 1]
    test_reg_models["LogisticRegression_distanceToGoal"] = {"model": LR_d, "val_preds": val_preds_reg_d,
                                                            "score_prob": score_prob_reg_d}
    test_poff_models["LogisticRegression_distanceToGoal"] = {"model": LR_d, "val_preds": val_preds_poff_d,
                                                             "score_prob": score_prob_poff_d}

    x_test_reg = df_test_regular["angleWithGoal"].to_numpy().reshape(-1, 1)
    x_test_poff = df_test_playoffs["angleWithGoal"].to_numpy().reshape(-1, 1)
    val_preds_reg_a = LR_a.predict(x_test_reg)
    val_preds_poff_a = LR_a.predict(x_test_poff)
    score_prob_reg_a = LR_a.predict_proba(x_test_reg)[:, 1]
    score_prob_poff_a = LR_a.predict_proba(x_test_poff)[:, 1]
    test_reg_models["LogisticRegression_angleWithGoal"] = {"model": LR_a, "val_preds": val_preds_reg_a,
                                                           "score_prob": score_prob_reg_a}
    test_poff_models["LogisticRegression_angleWithGoal"] = {"model": LR_a, "val_preds": val_preds_poff_a,
                                                            "score_prob": score_prob_poff_a}

    x_test_reg = df_test_regular[["distanceToGoal", "angleWithGoal"]].to_numpy().reshape(-1, 2)
    x_test_poff = df_test_playoffs[["distanceToGoal", "angleWithGoal"]].to_numpy().reshape(-1, 2)
    val_preds_reg_da = LR_da.predict(x_test_reg)
    val_preds_poff_da = LR_da.predict(x_test_poff)
    score_prob_reg_da = LR_da.predict_proba(x_test_reg)[:, 1]
    score_prob_poff_da = LR_da.predict_proba(x_test_poff)[:, 1]
    test_reg_models["LogisticRegression_distanceToGoal_angleWithGoal"] = {"model": LR_da, "val_preds": val_preds_reg_da,
                                                                          "score_prob": score_prob_reg_da}
    test_poff_models["LogisticRegression_distanceToGoal_angleWithGoal"] = {"model": LR_da,
                                                                           "val_preds": val_preds_poff_da,
                                                                           "score_prob": score_prob_poff_da}

    roc_auc_curve(y_test_reg, test_reg_models, plot=False, path_to_save="./figures/",
                  model_name="test_regular_baseline")
    goal_rate_curve(y_test_reg, test_reg_models, plot=False, path_to_save="./figures/",
                    model_name="test_regular_baseline")
    goal_rate_cumulative_curve(y_test_reg, test_reg_models, plot=False, path_to_save="./figures/",
                               model_name="test_regular_baseline")
    calibration(y_test_reg, test_reg_models, plot=False, path_to_save="./figures/", model_name="test_regular_baseline")

    roc_auc_curve(y_test_poff, test_poff_models, plot=False, path_to_save="./figures/",
                  model_name="test_playoffs_baseline")
    goal_rate_curve(y_test_poff, test_poff_models, plot=False, path_to_save="./figures/",
                    model_name="test_playoffs_baseline")
    goal_rate_cumulative_curve(y_test_poff, test_poff_models, plot=False, path_to_save="./figures/",
                               model_name="test_playoffs_baseline")
    calibration(y_test_poff, test_poff_models, plot=False, path_to_save="./figures/",
                model_name="test_playoffs_baseline")

    print("XGBoost simple model...")
    features = ['isGoal', 'distanceToGoal', 'angleWithGoal']
    (x, y, x_val, y_val), model, experiment = xgboost_model(df_train, features, 'XGBoost_distanceToGoal_angleWithGoal',
                                                            'xgboost_models', 'ift6758a-a22-g3-projet', comet=True)
    roc_auc_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/", model_name="simple_xgboost")
    goal_rate_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/", model_name="simple_xgboost")
    goal_rate_cumulative_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                               model_name="simple_xgboost")
    calibration(y_val, model, add_random=False, plot=False, path_to_save="./figures/", model_name="simple_xgboost")

    features = ['isGoal', 'speedOfChangeOfAngle', 'speed', 'changeOfAngleFromPrev', 'rebound', 'distanceFromPrev',
                'secondsSincePrev', 'prevAngleWithGoal', 'prevY', 'prevX', 'prevEvent', 'prevSecondsSinceStart',
                'angleWithGoal', 'distanceToGoal', 'x', 'y', 'emptyNet', 'strength', 'secondsSinceStart', 'shotType']

    df_mat = df_train[features]
    df_mat['strength'] = df_mat['strength'].fillna('Even')
    dummy_object = pd.get_dummies(df_mat[['strength', 'shotType', 'prevEvent']])
    df_mat = df_mat.merge(dummy_object, left_index=True, right_index=True)
    df_mat = df_mat.drop(labels=['strength', 'shotType', 'prevEvent'], axis=1)
    df_mat = df_mat.dropna(how='any')
    corr = df_mat.corr()
    fig, ax = plt.subplots(figsize=[23, 19])
    sns.heatmap(corr, center=0, cmap='coolwarm')
    plt.title("Matrice de corrélation des features générés")
    fig.savefig("./figures/correlation_matrix.png")
    plt.close()

    print("XGBoost all features model...")
    xgb_best_model = best_hyperparameters(method="random", n_iter=5)
    (x, y, x_val, y_val), model, experiment = xgboost_model(df_train, features, 'XGBoost_All_Features',
                                                            'xgboost_models', 'ift6758a-a22-g3-projet', xgb_best_model,
                                                            comet=True)
    roc_auc_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                  model_name="all_features_xgboost")
    goal_rate_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                    model_name="all_features_xgboost")
    goal_rate_cumulative_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                               model_name="all_features_xgboost")
    calibration(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                model_name="all_features_xgboost")

    print("XGBoost KBest features selection k=15, score_func=f_classif...")
    xgb_best_model = best_hyperparameters(method="random", n_iter=5)
    (x, y, x_val, y_val), model, experiment = xgboost_model(df_train, features, 'XGBoost_KBest_15_f_classif',
                                                            'xgboost_models', 'ift6758a-a22-g3-projet', xgb_best_model,
                                                            features_selection="k_best", score_func=f_classif, k=15,
                                                            comet=True)
    roc_auc_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                  model_name="kbest_15_f_classif_xgboost")
    goal_rate_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                    model_name="kbest_15_f_classif_xgboost")
    goal_rate_cumulative_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                               model_name="kbest_15_f_classif_xgboost")
    calibration(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                model_name="kbest_15_f_classif_xgboost")

    print("XGBoost KBest features selection k=25, score_func=f_classif...")
    xgb_best_model = best_hyperparameters(method="random", n_iter=5)
    (x, y, x_val, y_val), model, experiment = xgboost_model(df_train, features, 'XGBoost_KBest_25_f_classif',
                                                            'xgboost_models', 'ift6758a-a22-g3-projet', xgb_best_model,
                                                            features_selection="k_best", score_func=f_classif, k=25,
                                                            comet=True)
    roc_auc_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                  model_name="kbest_25_f_classif_xgboost")
    goal_rate_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                    model_name="kbest_25_f_classif_xgboost")
    goal_rate_cumulative_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                               model_name="kbest_25_f_classif_xgboost")
    calibration(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                model_name="kbest_25_f_classif_xgboost")

    print("XGBoost KBest features selection k=25, score_func=f_classif test...")
    xgb_model = model["XGBoost_KBest_25_f_classif"]["model"]
    kept_features = model["XGBoost_KBest_25_f_classif"]["features"]

    test_reg_models = {}
    test_poff_models = {}

    df_test_regular_filtered = filter_and_one_hot_encode_features(df_test_regular, features, 'R')
    y_test_reg = df_test_regular_filtered["isGoal"].to_numpy().reshape(-1)
    df_test_playoffs_filtered = filter_and_one_hot_encode_features(df_test_playoffs, features, 'P')
    # Add missing characteristics to playoffs
    df_test_playoffs_filtered['prevEvent_Goal'] = 0
    df_test_playoffs_filtered['prevEvent_Penalty'] = 0
    y_test_poff = df_test_playoffs_filtered["isGoal"].to_numpy().reshape(-1)

    x_test_reg = df_test_regular_filtered[kept_features].to_numpy().reshape(-1, 25)
    x_test_poff = df_test_playoffs_filtered[kept_features].to_numpy().reshape(-1, 25)
    val_preds_reg_d = xgb_model.predict(x_test_reg)
    val_preds_poff_d = xgb_model.predict(x_test_poff)
    score_prob_reg_d = xgb_model.predict_proba(x_test_reg)[:, 1]
    score_prob_poff_d = xgb_model.predict_proba(x_test_poff)[:, 1]
    test_reg_models["XGBoost_KBest_25_f_classif"] = {"model": xgb_model, "val_preds": val_preds_reg_d,
                                                            "score_prob": score_prob_reg_d}
    test_poff_models["XGBoost_KBest_25_f_classif"] = {"model": xgb_model, "val_preds": val_preds_poff_d,
                                                             "score_prob": score_prob_poff_d}

    roc_auc_curve(y_test_reg, test_reg_models, add_random=False, plot=False, path_to_save="./figures/",
                  model_name="test_regular_kbest_25_f_classif_xgboost")
    goal_rate_curve(y_test_reg, test_reg_models, add_random=False, plot=False, path_to_save="./figures/",
                    model_name="test_regular_kbest_25_f_classif_xgboost")
    goal_rate_cumulative_curve(y_test_reg, test_reg_models, add_random=False, plot=False, path_to_save="./figures/",
                               model_name="test_regular_kbest_25_f_classif_xgboost")
    calibration(y_test_reg, test_reg_models, add_random=False, plot=False, path_to_save="./figures/",
                model_name="test_regular_kbest_25_f_classif_xgboost")

    roc_auc_curve(y_test_poff, test_poff_models, add_random=False, plot=False, path_to_save="./figures/",
                  model_name="test_playoffs_kbest_25_f_classif_xgboost")
    goal_rate_curve(y_test_poff, test_poff_models, add_random=False, plot=False, path_to_save="./figures/",
                    model_name="test_playoffs_kbest_25_f_classif_xgboost")
    goal_rate_cumulative_curve(y_test_poff, test_poff_models, add_random=False, plot=False, path_to_save="./figures/",
                               model_name="test_playoffs_kbest_25_f_classif_xgboost")
    calibration(y_test_poff, test_poff_models, add_random=False, plot=False, path_to_save="./figures/",
                model_name="test_playoffs_kbest_25_f_classif_xgboost")

    print("XGBoost KBest features selection k=15, score_func=mutual_info_classif...")
    xgb_best_model = best_hyperparameters(method="random", n_iter=5)
    (x, y, x_val, y_val), model, experiment = xgboost_model(df_train, features, 'XGBoost_KBest_15_mutual_info_classif',
                                                            'xgboost_models', 'ift6758a-a22-g3-projet', xgb_best_model,
                                                            features_selection="k_best", score_func=mutual_info_classif,
                                                            k=15, comet=True)
    roc_auc_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                  model_name="kbest_15_mutual_info_classif_xgboost")
    goal_rate_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                    model_name="kbest_15_mutual_info_classif_xgboost")
    goal_rate_cumulative_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                               model_name="kbest_15_mutual_info_classif_xgboost")
    calibration(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                model_name="kbest_15_mutual_info_classif_xgboost")

    print("XGBoost KBest features selection k=25, score_func=mutual_info_classif...")
    xgb_best_model = best_hyperparameters(method="random", n_iter=5)
    (x, y, x_val, y_val), model, experiment = xgboost_model(df_train, features, 'XGBoost_KBest_25_mutual_info_classif',
                                                            'xgboost_models', 'ift6758a-a22-g3-projet', xgb_best_model,
                                                            features_selection="k_best", score_func=mutual_info_classif,
                                                            k=25, comet=True)
    roc_auc_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                  model_name="kbest_25_mutual_info_classif_xgboost")
    goal_rate_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                    model_name="kbest_25_mutual_info_classif_xgboost")
    goal_rate_cumulative_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                               model_name="kbest_25_mutual_info_classif_xgboost")
    calibration(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                model_name="kbest_25_mutual_info_classif_xgboost")

    print("XGBoost Recursive features selection...")
    (x, y, x_val, y_val), model, experiment = xgboost_model(df_train, features, 'XGBoost_Recursive', 'xgboost_models',
                                                            'ift6758a-a22-g3-projet', features_selection="recursive",
                                                            comet=True)
    roc_auc_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/", model_name="recursive_xgboost")
    goal_rate_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                    model_name="recursive_xgboost")
    goal_rate_cumulative_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                               model_name="recursive_xgboost")
    calibration(y_val, model, add_random=False, plot=False, path_to_save="./figures/", model_name="recursive_xgboost")
    fig, ax = plt.subplots(figsize=[14, 8])
    plt.plot(range(1, len(model["XGBoost_Recursive"]["model"].grid_scores_) + 1),
             model["XGBoost_Recursive"]["model"].grid_scores_)
    plt.title("Cross validation score according to the number of features, for the 5-fold")
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (accuracy)")
    plt.legend(["1st fold", "2nd fold", "3d fold", "4th fold", "5th fold"])
    fig.savefig("./figures/recursive_features_selection_cv_curve.png")
    plt.close()

    print("MLP model with Adam optimizer...")
    hyper_params = {
        "learning_rate": 0.0001,
        "batch_size": 50,
        "num_epochs": 2,
        "criterion": "BinaryCrossEntropy",
        "optimizer": "Adam"
    }
    net_adam = NetAdam()
    (x, y, x_val, y_val), model, experiment = mlp_model(df_train.copy(), features, 'MLP1', 'custom-models',
                                                        'ift6758a-a22-g3-projet', net_adam,
                                                        torch.optim.Adam(net_adam.parameters(),
                                                                         lr=hyper_params['learning_rate']),
                                                        hyper_params, comet=True)
    roc_auc_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                  model_name="MLP1")
    goal_rate_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                    model_name="MLP1")
    goal_rate_cumulative_curve(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                               model_name="MLP1")
    calibration(y_val, model, add_random=False, plot=False, path_to_save="./figures/",
                model_name="MLP1")

    print("MLP model with SGD optimizer...")
    hyper_params = {
        "learning_rate": 0.0003,
        "batch_size": 100,
        "num_epochs": 25,
        "momentum": 0.5,
        "criterion": "BinaryCrossEntropy",
        "optimizer": "SGD"
    }
    net_sgd = NetSGD()
    (x, y, x_val, y_val), model, experiment = mlp_model(df_train.copy(), features, 'MLP2', 'custom-models',
                                                        'ift6758a-a22-g3-projet', net_sgd,
                                                        torch.optim.SGD(net_sgd.parameters(),
                                                                        lr=hyper_params['learning_rate'],
                                                                        momentum=hyper_params['momentum']),
                                                        hyper_params, comet=True)

    print("k-NN model with 2 neighbors...")
    hyper_params = {
        "n_neighbors": 2
    }
    (x, y, x_val, y_val), model, experiment = knn_model(df_train.copy(), features, 'knn', 'custom-models',
                                                        'ift6758a-a22-g3-projet', hyper_params, comet=True)

    print('Evaluating MLP 1 model on test data set')
    download_model_from_comet("ift6758a-a22-g3-projet", "MLP1", "1.0.2", output_path="./models/")
    mlp_model = load_model_from_file('./models/MLP1.sav')

    df_test_regular_filtered = filter_and_one_hot_encode_features(df_test_regular, features, 'R')
    characteristics = list(df_test_regular_filtered.columns)
    characteristics.remove('isGoal')
    x_test, y_test = split_data_and_labels(df_test_regular_filtered, characteristics, ['isGoal'])
    accuracy, f1, result, values = evaluate_model(mlp_model, x_test, y_test)
    model = {}
    model["MLP1_Final_regular"] = {"values": values, "score_prob": result.numpy()}
    roc_auc_curve(y_test, model, add_random=False, plot=False, path_to_save="./figures/",
                  model_name="MLP1_Final_regular")
    goal_rate_curve(y_test, model, add_random=False, plot=False, path_to_save="./figures/",
                    model_name="MLP1_Final_regular")
    goal_rate_cumulative_curve(y_test, model, add_random=False, plot=False, path_to_save="./figures/",
                               model_name="MLP1_Final_regular")
    calibration(y_test, model, add_random=False, plot=False, path_to_save="./figures/",
                model_name="MLP1_Final_regular")

    df_test_playoffs_filtered = filter_and_one_hot_encode_features(df_test_playoffs, features, 'P')

    # Add missing characteristics to playoffs
    df_test_playoffs_filtered['prevEvent_Goal'] = 0
    df_test_playoffs_filtered['prevEvent_Penalty'] = 0

    characteristics = list(df_test_playoffs_filtered.columns)
    characteristics.remove('isGoal')
    x_test_playoffs, y_test_playoffs = split_data_and_labels(df_test_playoffs_filtered, characteristics, ['isGoal'])
    accuracy, f1, result_playoffs, values_playoffs = evaluate_model(mlp_model, x_test_playoffs, y_test_playoffs)
    model = {}
    model["MLP1_Final_playoffs"] = {"values": values_playoffs, "score_prob": result_playoffs.numpy()}
    roc_auc_curve(y_test_playoffs, model, add_random=False, plot=False, path_to_save="./figures/",
                  model_name="MLP1_Final_playoffs")
    goal_rate_curve(y_test_playoffs, model, add_random=False, plot=False, path_to_save="./figures/",
                    model_name="MLP1_Final_playoffs")
    goal_rate_cumulative_curve(y_test_playoffs, model, add_random=False, plot=False, path_to_save="./figures/",
                               model_name="MLP1_Final_playoffs")
    calibration(y_test_playoffs, model, add_random=False, plot=False, path_to_save="./figures/",
                model_name="MLP1_Final_playoffs")
