import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    KFold,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    recall_score,
    confusion_matrix,
    f1_score
)

from sklearn.utils import resample

from . import Analysis

def RandomForest(self, verbose=True, output_plots=True):
    """
    Hyperparameter search for Random Forest model, returns optimal hyperparameters
    """

    if verbose:
        print("RandomForest from RF.py is working!")
    #N.B. Do not have to include feature scaling as a preprocessing step - it has no effect for tree classifiers
    forest = RandomForestClassifier(n_jobs=-1, random_state=0) #n_jobs = number of jobs to run in parallel, -1 means use all processors
    param_grid = {'n_estimators':[10,20,30,40,50,60,100], 'max_depth':[2,4,6,8,10,12,14,16]}

    #Search for best parameters for accuracy
    if self.scoring_metric == "accuracy":

        #Set up dictionary for scoring results
        scores = {
            "df_name": [],
            "search_algorithm": [],
            "best_estimator": [],
            "cross_validation_score": [],
            "accuracy_score": []
        }

        for key, df in self.input_df.items():

            #Grid search with stratified kfold cross validation (default for classifiers)
            skfold = StratifiedKFold(n_splits=5)
            grid_strat = GridSearchCV(forest, param_grid, cv=skfold, scoring='accuracy', refit=True)

            grid_strat.fit(self.X_train_nd[key], self.y_train_nd[key])

            scores["df_name"].append(key)
            scores["search_algorithm"].append("Stratified grid search")
            scores["best_estimator"].append(grid_strat.best_estimator_)
            scores["cross_validation_score"].append(grid_strat.best_score_)
            scores["accuracy_score"].append(accuracy_score(self.y_train_nd[key], grid_strat.predict(self.X_train_nd[key])))

            #Grid search with kfold cross validation
            kfold = KFold(n_splits=5)
            grid_kfold = GridSearchCV(forest, param_grid, cv=kfold, scoring='accuracy', refit=True)
            grid_kfold.fit(self.X_train_nd[key], self.y_train_nd[key])

            scores["df_name"].append(key)
            scores["search_algorithm"].append("Kfold grid search")
            scores["best_estimator"].append(grid_kfold.best_estimator_)
            scores["cross_validation_score"].append(grid_kfold.best_score_)
            scores["accuracy_score"].append(accuracy_score(self.y_train_nd[key], grid_kfold.predict(self.X_train_nd[key])))

            #Search with shuffled cross validation
            kfold_shuffle=KFold(n_splits=5, shuffle=True, random_state=0)
            grid_kfold_shuffle = GridSearchCV(forest, param_grid, cv=kfold_shuffle, scoring='accuracy', refit=True)
            grid_kfold_shuffle.fit(self.X_train_nd[key], self.y_train_nd[key])

            scores["df_name"].append(key)
            scores["search_algorithm"].append("Grid kfold shuffle search")
            scores["best_estimator"].append(grid_kfold_shuffle.best_estimator_)
            scores["cross_validation_score"].append(grid_kfold_shuffle.best_score_)
            scores["accuracy_score"].append(accuracy_score(self.y_train_nd[key], grid_kfold_shuffle.predict(self.X_train_nd[key])))

        if verbose:
            print(scores)

        #Identify max score (accuracy or cross-validation) and store in new dictionary
        #Create new dictionary with only floats and convert to array
        scores_subset = np.array([scores["cross_validation_score"]])
        if verbose:
            print("CV scores:", scores_subset)

        #Obtain max value and position of max
        i,j = np.unravel_index(scores_subset.argmax(), scores_subset.shape)
        if verbose:
            print("Max CV score: ", scores_subset[i,j])
            print("Position of max:", i, j)

        #Obtain value in another row with the same column as the max
        if verbose:
            print("Features for max:", list(scores["df_name"])[j])
            print("Algorithm for max:", list(scores["search_algorithm"])[j])
            print("Estimator for max:", list(scores["best_estimator"])[j])
            print("CV score for max:", list(scores["cross_validation_score"])[j])
            print("Accuracy score for max:", list(scores["accuracy_score"])[j])

        #Rename max parameters for input to instance(?) methods
        opt_features = list(scores["df_name"])[j]
        if verbose:
            print(opt_features)
        opt_algorithm = list(scores["search_algorithm"])[j]
        if verbose:
            print(opt_algorithm)
        opt_estimator = list(scores["best_estimator"])[j]
        if verbose:
            print(opt_estimator)
        max_cv_score = list(scores["cross_validation_score"])[j]
        if verbose:
            print(max_cv_score)
        max_acc_score = list(scores["accuracy_score"])[j]
        if verbose:
            print(max_acc_score)

        #Retrains model with optimal parameters
        opt_estimator.fit(self.X_train_nd[opt_features], self.y_train_nd[opt_features])
        #tested to see whether fit above runs through correctly to next .py - it does

        #Attach variables to self so that they can be accessed in Models.py
        self.opt_estimator = opt_estimator
        self.opt_features = opt_features

        #Plot feature importances
        if output_plots:
            Analysis.plot_feature_importances(self, opt_estimator, opt_features)

        ###Generate confusion matrix and metrics with optimal parameters for TEST SET
        if output_plots:
            Analysis.metrics_and_confusion_on_test_set(self, opt_estimator, opt_features)

        ###Generate confusion matrix and metrics with optimal parameters for TRAINING SET
        if output_plots:
            Analysis.metrics_on_training_set(self, opt_estimator, opt_features)

    #Searches for best parameters for precision
    elif self.scoring_metric == "average_precision":
        #Set up dictionary for scoring results
        scores = {
            "df_name": [],
            "search_algorithm": [],
            "best_estimator": [],
            "cross_validation_score": [],
            "average_precision_score": []
        }

        for key, df in self.input_df.items():

        #use predict_proba function for positive class to calculate average_precision_score
            #Grid search with stratified cross validation
            skfold = StratifiedKFold(n_splits=5)
            grid_strat = GridSearchCV(forest, param_grid, cv=skfold, scoring='average_precision', refit=True)
            grid_strat.fit(self.X_train_nd[key], self.y_train_nd[key])

            scores["df_name"].append(key)
            scores["search_algorithm"].append("Stratified grid search")
            scores["best_estimator"].append(grid_strat.best_estimator_)
            scores["cross_validation_score"].append(grid_strat.best_score_)
            scores["average_precision_score"].append(average_precision_score(self.y_train_nd[key], grid_strat.predict_proba(self.X_train_nd[key])[:,1]))
            #Use predict_proba as RandomForestClassifier does not have a decision_function method

            #Grid search with kfold cross validation
            kfold = KFold(n_splits=5)
            grid_kfold = GridSearchCV(forest, param_grid, cv=kfold, scoring='average_precision', refit=True)
            grid_kfold.fit(self.X_train_nd[key], self.y_train_nd[key])

            scores["df_name"].append(key)
            scores["search_algorithm"].append("Kfold grid search")
            scores["best_estimator"].append(grid_kfold.best_estimator_)
            scores["cross_validation_score"].append(grid_kfold.best_score_)
            scores["average_precision_score"].append(average_precision_score(self.y_train_nd[key], grid_kfold.predict_proba(self.X_train_nd[key])[:,1]))

            #Search with shuffled cross validation
            kfold_shuffle=KFold(n_splits=5, shuffle=True, random_state=0)
            grid_kfold_shuffle = GridSearchCV(forest, param_grid, cv=kfold_shuffle, scoring='average_precision', refit=True)
            grid_kfold_shuffle.fit(self.X_train_nd[key], self.y_train_nd[key])

            scores["df_name"].append(key)
            scores["search_algorithm"].append("Grid kfold shuffle search")
            scores["best_estimator"].append(grid_kfold_shuffle.best_estimator_)
            scores["cross_validation_score"].append(grid_kfold_shuffle.best_score_)
            scores["average_precision_score"].append(average_precision_score(self.y_train_nd[key], grid_kfold_shuffle.predict_proba(self.X_train_nd[key])[:,1]))

            #print("with predict_proba: ", grid_kfold_shuffle.predict_proba(self.X_test_nd[key])[:,1])
            #print("with predict: ", grid_kfold_shuffle.predict(self.X_test_nd[key]))

        if verbose:
            print(scores)

        #Identify max score (av precision or cross-validation) and store in new dictionary
        #Create new dictionary with only floats and convert to array
        scores_subset = np.array([scores["cross_validation_score"]])
        if verbose:
            print("CV scores:", scores_subset)

        #Obtain max value and position of max
        i,j = np.unravel_index(scores_subset.argmax(), scores_subset.shape)
        if verbose:
            print("Max CV score:",scores_subset[i,j])
            print("Position of max:", i, j)

        #Obtain value in another row with the same column as the max
        if verbose:
            print("Features for max:", list(scores["df_name"])[j])
            print("Algorithm for max:", list(scores["search_algorithm"])[j])
            print("Estimator for max:", list(scores["best_estimator"])[j])
            print("CV score for max:", list(scores["cross_validation_score"])[j])
            print("Average precision score for max:", list(scores["average_precision_score"])[j])

        ###Generate precision_recall_curve with max parameters
        opt_features = list(scores["df_name"])[j]
        #if verbose: print(opt_features)
        opt_algorithm = list(scores["search_algorithm"])[j]
        #if verbose: print(opt_algorithm)
        opt_estimator = list(scores["best_estimator"])[j]
        #if verbose: print(opt_estimator)
        max_cv_score = list(scores["cross_validation_score"])[j]
        #if verbose: print(max_cv_score)
        max_av_precision_score = list(scores["average_precision_score"])[j]
        #if verbose: print(max_av_precision_score)


        #Retrains model with optimal parameters
        opt_estimator.fit(self.X_train_nd[opt_features], self.y_train_nd[opt_features])

        #Attach variables to self so that they can be accessed in Models.py
        self.opt_estimator = opt_estimator
        self.opt_features = opt_features

        ###Plot precision_recall_curve on training set for optimal hyperparameters and shows point for best F1 score
        Analysis.pr_curve_on_training_set_for_all(self, opt_estimator, opt_features,verbose=verbose, output_plots=output_plots) #N.B. Includes metrics with shifted thresholds

        #Plot feature importances
        if output_plots:
            Analysis.plot_feature_importances(self, opt_estimator, opt_features)

        ###Generate confusion matrix and metrics with optimal parameters for TEST SET
        if output_plots:
            Analysis.metrics_and_confusion_on_test_set(self, opt_estimator, opt_features)

        ###Generate confusion matrix and metrics with optimal parameters for TRAINING SET
        if output_plots:
            Analysis.metrics_on_training_set(self, opt_estimator, opt_features)

    #Searches for best parameters for ROC AUC
    #Uses predict_proba as RF doesn't have a decision function method
    elif self.scoring_metric == "roc_auc":
        #Set up dictionary for scoring results
        scores = {
            "df_name": [],
            "search_algorithm": [],
            "best_estimator": [],
            "cross_validation_score": [],
            "roc_auc_score": []
        }

        for key, df in self.input_df.items():
        #Grid search with stratified cross validation
        #use decision function or predict_proba to calculate roc_auc
            skfold = StratifiedKFold(n_splits=5)
            grid_strat = GridSearchCV(forest, param_grid, cv=skfold, scoring='roc_auc', refit=True)
            grid_strat.fit(self.X_train_nd[key], self.y_train_nd[key])

            scores["df_name"].append(key)
            scores["search_algorithm"].append("Stratified grid search")
            scores["best_estimator"].append(grid_strat.best_estimator_)
            scores["cross_validation_score"].append(grid_strat.best_score_)
            scores["roc_auc_score"].append(roc_auc_score(self.y_train_nd[key], grid_strat.predict_proba(self.X_train_nd[key])[:,1]))

            #Grid search with kfold cross validation
            kfold = KFold(n_splits=5)
            grid_kfold = GridSearchCV(forest, param_grid, cv=kfold, scoring='roc_auc', refit=True)
            grid_kfold.fit(self.X_train_nd[key], self.y_train_nd[key])

            scores["df_name"].append(key)
            scores["search_algorithm"].append("Kfold grid search")
            scores["best_estimator"].append(grid_kfold.best_estimator_)
            scores["cross_validation_score"].append(grid_kfold.best_score_)
            scores["roc_auc_score"].append(roc_auc_score(self.y_train_nd[key], grid_kfold.predict_proba(self.X_train_nd[key])[:,1]))

            #Search with shuffled cross validation
            kfold_shuffle=KFold(n_splits=5, shuffle=True, random_state=0)
            grid_kfold_shuffle = GridSearchCV(forest, param_grid, cv=kfold_shuffle, scoring='roc_auc', refit=True)
            grid_kfold_shuffle.fit(self.X_train_nd[key], self.y_train_nd[key])

            scores["df_name"].append(key)
            scores["search_algorithm"].append("Grid kfold shuffle search")
            scores["best_estimator"].append(grid_kfold_shuffle.best_estimator_)
            scores["cross_validation_score"].append(grid_kfold_shuffle.best_score_)
            scores["roc_auc_score"].append(roc_auc_score(self.y_train_nd[key], grid_kfold_shuffle.predict_proba(self.X_train_nd[key])[:,1]))

        if verbose:
            print(scores)

        #Identify max score (roc_auc_score or cross-validation) and store in new dictionary
        #Create new dictionary with only floats and convert to array
        scores_subset = np.array([scores["cross_validation_score"]])
        if verbose:
            print("CV scores:", scores_subset)

        #Obtain max value and position of max
        i,j = np.unravel_index(scores_subset.argmax(), scores_subset.shape)
        if verbose:
            print("Max CV score:", scores_subset[i,j])
            print("Position of max:", i, j)

        #Obtain value in another row with the same column as the max
        if verbose:
            print("Features for max:", list(scores["df_name"])[j])
            print("Algorithm for max:", list(scores["search_algorithm"])[j])
            print("Estimator for max:", list(scores["best_estimator"])[j])
            print("CV score for max:", list(scores["cross_validation_score"])[j])
            print("ROC AUC score for max:", list(scores["roc_auc_score"])[j])

        ###Generate ROC curve with max parameters
        opt_features = list(scores["df_name"])[j]
        #if verbose: print(opt_features)
        opt_algorithm = list(scores["search_algorithm"])[j]
        #if verbose: print(opt_algorithm)
        opt_estimator = list(scores["best_estimator"])[j]
        #if verbose: print(opt_estimator)
        max_cv_score = list(scores["cross_validation_score"])[j]
        #if verbose: print(max_cv_score)
        max_roc_auc_score = list(scores["roc_auc_score"])[j]
        #if verbose: print(max_roc_auc_score)

        #Reruns ML model with optimal hyperparameters
        opt_estimator.fit(self.X_train_nd[opt_features], self.y_train_nd[opt_features])

        #Attach variables to self so that they can be accessed in Models.py
        self.opt_estimator = opt_estimator
        self.opt_features = opt_features

        #Plot ROC curve for training set and calculate ROC_AUC for training and test sets
        if output_plots:
            Analysis.roc_curve_on_training_set_with_proba(self, opt_estimator, opt_features)

        #Plot feature importances
        if output_plots:
            Analysis.plot_feature_importances(self, opt_estimator, opt_features)

        ###Generate confusion matrix and metrics with optimal parameters for TEST SET (from Analysis.py)
        if output_plots:
            Analysis.metrics_and_confusion_on_test_set(self, opt_estimator, opt_features)

        ###Generate confusion matrix and metrics with optimal parameters for TRAINING SET (from Analysis.py)
        if output_plots:
            Analysis.metrics_on_training_set(self, opt_estimator, opt_features)

    #Searches for best parameters for recall
    elif self.scoring_metric == "recall":
        #Set up dictionary for scoring results
        scores = {
            "df_name": [],
            "search_algorithm": [],
            "best_estimator": [],
            "cross_validation_score": [],
            "recall_score": []
        }

        for key, df in self.input_df.items():
            #Grid search with stratified cross validation
            skfold = StratifiedKFold(n_splits=5)
            grid_strat = GridSearchCV(forest, param_grid, cv=skfold, scoring='recall', refit=True)
            grid_strat.fit(self.X_train_nd[key], self.y_train_nd[key])

            scores["df_name"].append(key)
            scores["search_algorithm"].append("Stratified grid search")
            scores["best_estimator"].append(grid_strat.best_estimator_)
            scores["cross_validation_score"].append(grid_strat.best_score_)
            scores["recall_score"].append(recall_score(self.y_train_nd[key], grid_strat.predict(self.X_train_nd[key])))

            #Grid search with kfold cross validation
            kfold = KFold(n_splits=5)
            grid_kfold = GridSearchCV(forest, param_grid, cv=kfold, scoring='recall', refit=True)
            grid_kfold.fit(self.X_train_nd[key], self.y_train_nd[key])

            scores["df_name"].append(key)
            scores["search_algorithm"].append("Kfold grid search")
            scores["best_estimator"].append(grid_kfold.best_estimator_)
            scores["cross_validation_score"].append(grid_kfold.best_score_)
            scores["recall_score"].append(recall_score(self.y_train_nd[key], grid_kfold.predict(self.X_train_nd[key])))

            #Search with shuffled cross validation
            kfold_shuffle=KFold(n_splits=5, shuffle=True, random_state=0)
            grid_kfold_shuffle = GridSearchCV(forest, param_grid, cv=kfold_shuffle, scoring='recall', refit=True)
            grid_kfold_shuffle.fit(self.X_train_nd[key], self.y_train_nd[key])

            scores["df_name"].append(key)
            scores["search_algorithm"].append("Grid kfold shuffle search")
            scores["best_estimator"].append(grid_kfold_shuffle.best_estimator_)
            scores["cross_validation_score"].append(grid_kfold_shuffle.best_score_)
            scores["recall_score"].append(recall_score(self.y_train_nd[key], grid_kfold_shuffle.predict(self.X_train_nd[key])))

        if verbose:
            print(scores)

        #Identify max score (recall or cross-validation) and store in new dictionary
        #Create new dictionary with only floats and convert to array
        scores_subset = np.array([scores["cross_validation_score"]])
        if verbose:
            print("CV scores:", scores_subset)

        #Obtain max value and position of max
        i,j = np.unravel_index(scores_subset.argmax(), scores_subset.shape)
        if verbose:
            print("Max CV score: ", scores_subset[i,j])
            print("Position of max:", i, j)

        #Obtain value in another row with the same column as the max
        if verbose:
            print("Features for max:", list(scores["df_name"])[j])
            print("Algorithm for max:", list(scores["search_algorithm"])[j])
            print("Estimator for max:", list(scores["best_estimator"])[j])
            print("CV score for max:", list(scores["cross_validation_score"])[j])
            print("Recall score for max:", list(scores["recall_score"])[j])

        #Rename max parameters for input to instance(?) methods
        opt_features = list(scores["df_name"])[j]
        if verbose:
            print(opt_features)
        opt_algorithm = list(scores["search_algorithm"])[j]
        if verbose:
            print(opt_algorithm)
        opt_estimator = list(scores["best_estimator"])[j]
        if verbose:
            print(opt_estimator)
        max_cv_score = list(scores["cross_validation_score"])[j]
        if verbose:
            print(max_cv_score)
        max_recall_score = list(scores["recall_score"])[j]
        if verbose:
            print(max_recall_score)

        #Retrains model with optimal parameters
        opt_estimator.fit(self.X_train_nd[opt_features], self.y_train_nd[opt_features])
        #tested to see whether fit above runs through correctly to next .py - it does

        #Attach variables to self so that they can be accessed in Models.py
        self.opt_estimator = opt_estimator
        self.opt_features = opt_features

        Analysis.pr_curve_on_training_set_for_all(self, opt_estimator, opt_features,verbose=verbose, output_plots=output_plots) 

        #Plot feature importances
        if output_plots:
            Analysis.plot_feature_importances(self, opt_estimator, opt_features)

        ###Generate confusion matrix and metrics with optimal parameters for TEST SET
        if output_plots:
            Analysis.metrics_and_confusion_on_test_set(self, opt_estimator, opt_features)

        ###Generate confusion matrix and metrics with optimal parameters for TRAINING SET
        if output_plots:
            Analysis.metrics_on_training_set(self, opt_estimator, opt_features)
