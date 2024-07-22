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
    confusion_matrix,
    f1_score
)

"""
Set of instance methods for running ML models with optimal hyperparameters, then generating metrics and plots for different models
"""

###Generate confusion matrix and metrics with optimal parameters for TEST SET
def metrics_and_confusion_on_test_set(self, opt_estimator, opt_features, verbose=True, output_plots=True):
    if verbose:
        print("metrics_on_test_set from Analysis.py is working!")
        print("opt_estimator is ", opt_estimator)
        print("opt_features are ", opt_features)

    confusion = confusion_matrix(self.y_test_nd[opt_features], opt_estimator.predict(self.X_test_nd[opt_features]))#.astype(int))

    #Calculate metrics
    Precision = (confusion[1][1]) / (confusion[1][1] + confusion[0][1]) if (confusion[1][1] + confusion[0][1]) != 0 else 0
    Sensitivity = (confusion[1][1])/(confusion[1][1]+confusion[1][0])
    Specificity = (confusion[0][0])/(confusion[0][0]+confusion[0][1])
    FPR = 1-Specificity
    F1_Score = f1_score(self.y_test_nd[opt_features], opt_estimator.predict(self.X_test_nd[opt_features]))
    number_actual_test_zeros = len(self.y_test_nd[opt_features]) - np.count_nonzero(self.y_test_nd[opt_features])
    if verbose:
        print("number of actual TEST SET zeros", number_actual_test_zeros)
    very_major_error = (confusion[0][1]/number_actual_test_zeros)*100
    number_actual_test_ones = np.count_nonzero(self.y_test_nd[opt_features])
    if verbose:
        print("number of actual TEST SET ones", number_actual_test_ones)
    major_error = (confusion[1][0]/number_actual_test_ones)*100
    roc_auc = roc_auc_score(self.y_test_nd[opt_features], opt_estimator.predict_proba(self.X_test_nd[opt_features])[:,1])

    if verbose:
        print('Precision for TEST set:', Precision)
        print('Sensitivity for TEST set:', Sensitivity)
        print('Specificity for TEST set:', Specificity)
        print('F1 Score for TEST set:', F1_Score)
        print('FPR for TEST set:', FPR)
        print('very major error for TEST set:', very_major_error)
        print('major error for TEST set: ', major_error)
        print('\n confusion matrix for TEST set: \n', confusion)
        print('ROC_AUC for TEST set:', roc_auc)

    test_set_metrics = {"Precision": Precision, "Sensitivity": Sensitivity, "Specificity": Specificity, "F1_Score": F1_Score, "FPR": FPR, "VME": very_major_error, "ME": major_error, "Confusion_matrix": confusion, "ROC_AUC": roc_auc}

    #Plot confusion matrix - help from https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
    if output_plots:
        plt.figure()
        names = ['True R','False S','False R','True S']
        counts = ["{0:0.0f}".format(value) for value in confusion.flatten()]
        percentages = ["{0:.1%}".format(value) for value in confusion.flatten()/np.sum(confusion)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(names,counts,percentages)]
        labels = np.asarray(labels).reshape(2,2)

        sns.heatmap(confusion, annot=labels, fmt="", cmap="PuBu", cbar=False)
        plt.xlabel("Predicted", fontsize = 12) #Must be placed after heatmap generation
        plt.ylabel("True", fontsize = 12)
        plt.title("Confusion matrix on TEST set, NO shifted_threshold")

    return test_set_metrics

###Generate confusion matrix and metrics with optimal parameters for TRAINING SET
def metrics_on_training_set(self, opt_estimator, opt_features, verbose=True):
    if verbose:
        print("metrics_on_training_set from Analysis.py is working!")
        print("opt_estimator is ", opt_estimator)
        print("opt_features are ", opt_features)

    confusion_train = confusion_matrix(self.y_train_nd[opt_features], opt_estimator.predict(self.X_train_nd[opt_features]).astype(int))

    #Calculate metrics
    Precision_train = (confusion_train[1][1])/(confusion_train[1][1]+confusion_train[0][1])
    Sensitivity_train = (confusion_train[1][1])/(confusion_train[1][1]+confusion_train[1][0])
    Specificity_train = (confusion_train[0][0])/(confusion_train[0][0]+confusion_train[0][1])
    F1_Score_train = f1_score(self.y_train_nd[opt_features], opt_estimator.predict(self.X_train_nd[opt_features]))
    FPR_train = 1-Specificity_train
    number_actual_test_zeros_train = len(self.y_train_nd[opt_features]) - np.count_nonzero(self.y_train_nd[opt_features])
    if verbose:
        print("number of actual TRAINING SET zeros", number_actual_test_zeros_train)
    very_major_error_train = (confusion_train[0][1]/number_actual_test_zeros_train)*100
    number_actual_test_ones_train = np.count_nonzero(self.y_train_nd[opt_features])
    if verbose:
        print("number of actual TRAINING SET ones", number_actual_test_ones_train)
    major_error_train = (confusion_train[1][0]/number_actual_test_ones_train)*100
    roc_auc_train = roc_auc_score(self.y_train_nd[opt_features], opt_estimator.predict_proba(self.X_train_nd[opt_features])[:,1])

    if verbose:
        print('Precision for TRAINING set:', Precision_train)
        print('Sensitivity for TRAINING set:', Sensitivity_train)
        print('Specificity for TRAINING set:', Specificity_train)
        print('F1 Score for TRAINING set:', F1_Score_train)
        print('FPR for TRAINING set:', FPR_train)
        print('very major error for TRAINING set:', very_major_error_train)
        print('major error for TRAINING set: ', major_error_train)
        print('\n confusion matrix for TRAINING set: \n', confusion_train)
        print('ROC_AUC for TRAINING set:', roc_auc_train)


#Plot feature coefficients for linear models
def plot_feature_coeffs(self, opt_estimator, opt_features, verbose=True):
    plt.figure()
    coeffs = opt_estimator.named_steps["classifier"].coef_[0]
    if verbose:
        print("These are the feature coeffs:", coeffs)
        print("Type:", type(coeffs))
    n_features = self.input_df[opt_features].columns[:-1]
    if verbose:
        print("Features for feature coeff plt:", n_features)
        print("Number of features during fit:", opt_estimator.named_steps["classifier"].n_features_in_)
        print("Names of features during fit:", opt_estimator.get_feature_names_out)
    fig = plt.barh(n_features, coeffs, align='center') #barh(y, width, etc.)
    plt.xlabel('feature coeff')
    plt.ylabel('feature')
    plt.yticks(fontsize = 8)
    plt.title('feature coefficient plot')
    plt.tight_layout()

#Plot feature importances for RF, DT and GBDT models
def plot_feature_importances(self, opt_estimator, opt_features, verbose=True):
    plt.figure()
    n_features = self.input_df[opt_features].columns[:-1]
    if verbose:
        print("Features for feature plt:", n_features)
        print("These are the feature importances:", opt_estimator.feature_importances_)
    fig = plt.barh(n_features, opt_estimator.feature_importances_, align='center') #barh(y, width, etc.)
    plt.xlabel('feature importance')
    plt.ylabel('feature')
    plt.yticks(fontsize = 8)
    plt.title('feature importance plot')
    plt.tight_layout()

###Generate PR curves for TRAINING SET with threshold set at highest F1 score
###Calculations done using predict_proba not decision_function, therefore
###this method works for BOTH TREE MODELS (i.e. those which don't have a
###decision_function method) AND LINEAR MODELS (which do have a
###decision_function method)
### I have checked that the same results are produced for LR with my old decision_function method and the new predict_proba method below

def pr_curve_on_training_set_for_all(self, opt_estimator, opt_features, verbose=True, output_plots=True):
    if verbose:
        print("pr_curve_on_training_set_for_all from Analysis.py is working!")
        print("opt_estimator is ", opt_estimator)
        print("opt_features are ", opt_features)

    #Calculate proba threshold with highest F1 score
    def proba_to_labels(proba_pos, threshold):
        return (proba_pos >= threshold).astype('int')

    y_scores_predict_proba = opt_estimator.predict_proba(self.X_train_nd[opt_features])
    y_scores_predict_proba_pos = y_scores_predict_proba[:, 1]

    precision_proba, recall_proba, thresholds_proba = precision_recall_curve(self.y_train_nd[opt_features], y_scores_predict_proba_pos)
    if verbose:
        print("thresholds_proba:", thresholds_proba)

    f1_scores_for_proba = [f1_score(self.y_train_nd[opt_features], proba_to_labels(y_scores_predict_proba_pos, t_proba)) for t_proba in thresholds_proba]
    if verbose:
        print("f1_scores for each threshold from proba: ", f1_scores_for_proba, len(f1_scores_for_proba))
    max_f1_score_index_for_proba = np.argmax(f1_scores_for_proba) #returns index for max f1_score
    if verbose:
        print("max_f1_score_index_for_proba", max_f1_score_index_for_proba)

    #Get opt_threshold in terms of proba_pos
    opt_threshold = thresholds_proba[max_f1_score_index_for_proba]
    if verbose:
        print("opt_threshold:", opt_threshold)

    #Attach variable to self so that it can be accessed in Models.py
    self.opt_threshold = opt_threshold

    #Get default threshold - i.e. point closest to a threshold_proba of 0.5
    close_zero = np.argmin(np.abs(thresholds_proba-0.5)) #comes from proba - selects point closest to 0.5
    if verbose:
        print("close_zero:", close_zero)

    if output_plots:
        plt.figure()
        plt.plot(recall_proba, precision_proba, label='PR curve')
        plt.plot(recall_proba[close_zero], precision_proba[close_zero], '^', c='k', markersize=10, label='default threshold', fillstyle='none', mew=2)
        plt.plot(recall_proba[max_f1_score_index_for_proba], precision_proba[max_f1_score_index_for_proba], '^', label='threshold for max F1')
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.legend(loc='best')
        plt.title("Precision-recall curve for TRAINING SET with  "+str(opt_features)+" features and "+str(opt_estimator))

    #Generate confusion matrix and metrics with shifted threshold for TRAINING SET
    metrics_on_training_set_shiftedthres(self, opt_estimator, opt_features, opt_threshold, verbose=verbose)

    #Generate confusion matrix and metrics with shifted threshold for TEST SET
    metrics_on_test_set_shiftedthres(self, opt_estimator, opt_features, opt_threshold, verbose=verbose, output_plots=output_plots)


###Generate confusion matrix and metrics with shifted threshold for TRAINING SET
def metrics_on_training_set_shiftedthres(self, opt_estimator, opt_features, opt_threshold, verbose=True):
    if verbose:
        print("metrics_on_training_set_shiftedthres from Analysis.py is working!")
        print("opt_estimator for opt_threshold is ", opt_estimator)
        print("opt_features for opt_threshold are ", opt_features)
        print("opt_threshold is", opt_threshold)

    new_predicted_y_train = (opt_estimator.predict_proba(self.X_train_nd[opt_features])[:,1] >= opt_threshold).astype(int)
    if verbose:
        print("new_predicted_y_train:", new_predicted_y_train)

    confusion_train_shifted = confusion_matrix(self.y_train_nd[opt_features], new_predicted_y_train)

    Precision_train_shifted = (confusion_train_shifted[1][1])/(confusion_train_shifted[1][1]+confusion_train_shifted[0][1])
    Sensitivity_train_shifted = (confusion_train_shifted[1][1])/(confusion_train_shifted[1][1]+confusion_train_shifted[1][0])
    Specificity_train_shifted = (confusion_train_shifted[0][0])/(confusion_train_shifted[0][0]+confusion_train_shifted[0][1])
    F1_Score_train_shifted = f1_score(self.y_train_nd[opt_features], new_predicted_y_train)
    FPR_train_shifted = 1-Specificity_train_shifted
    number_actual_test_zeros_train = len(self.y_train_nd[opt_features]) - np.count_nonzero(self.y_train_nd[opt_features])
    if verbose:
        print("number of actual TRAINING SET zeros", number_actual_test_zeros_train)
    very_major_error_train_shifted = (confusion_train_shifted[0][1]/number_actual_test_zeros_train)*100
    number_actual_test_ones_train = np.count_nonzero(self.y_train_nd[opt_features])
    if verbose:
        print("number of actual TRAINING SET ones", number_actual_test_ones_train)
    major_error_train_shifted = (confusion_train_shifted[1][0]/number_actual_test_ones_train)*100

    if verbose:
        print('Precision for TRAINING set _shifted:', Precision_train_shifted)
        print('Sensitivity for TRAINING set _shifted:', Sensitivity_train_shifted)
        print('Specificity for TRAINING set _shifted:', Specificity_train_shifted)
        print('F1 Score for TRAINING set _shifted:', F1_Score_train_shifted)
        print('FPR for TRAINING set _shifted:', FPR_train_shifted)
        print('very major error for TRAINING set _shifted:', very_major_error_train_shifted)
        print('major error for TRAINING set _shifted: ', major_error_train_shifted)
        print('\n confusion matrix for TRAINING set _shifted: \n', confusion_train_shifted)

###Generate confusion matrix and metrics with shifted threshold for TEST SET
def metrics_on_test_set_shiftedthres(self, opt_estimator, opt_features, opt_threshold, verbose=True, output_plots=True):
    if verbose:
        print("metrics_on_test_set_shiftedthres from Analysis.py is working!")
        print("opt_estimator for opt_threshold is ", opt_estimator)
        print("opt_features for opt_threshold are ", opt_features)
        print("opt_threshold is", opt_threshold)

    new_predicted_y_test = (opt_estimator.predict_proba(self.X_test_nd[opt_features])[:,1] >= opt_threshold).astype(int)
    if verbose:
        print("new_predicted_y_test:", new_predicted_y_test)

    confusion_test_shifted = confusion_matrix(self.y_test_nd[opt_features], new_predicted_y_test)

    #Calculate metrics
    Precision_test_shifted = (confusion_test_shifted[1][1])/(confusion_test_shifted[1][1]+confusion_test_shifted[0][1])
    Sensitivity_test_shifted = (confusion_test_shifted[1][1])/(confusion_test_shifted[1][1]+confusion_test_shifted[1][0])
    Specificity_test_shifted = (confusion_test_shifted[0][0])/(confusion_test_shifted[0][0]+confusion_test_shifted[0][1])
    F1_Score_test_shifted = f1_score(self.y_test_nd[opt_features], new_predicted_y_test)
    FPR_test_shifted = 1-Specificity_test_shifted
    number_actual_test_zeros = len(self.y_test_nd[opt_features]) - np.count_nonzero(self.y_test_nd[opt_features])
    if verbose:
        print("number of actual TEST SET zeros", number_actual_test_zeros)
    very_major_error_test_shifted = (confusion_test_shifted[0][1]/number_actual_test_zeros)*100
    number_actual_test_ones = np.count_nonzero(self.y_test_nd[opt_features])
    if verbose:
        print("number of actual TEST SET ones", number_actual_test_ones)
    major_error_test_shifted = (confusion_test_shifted[1][0]/number_actual_test_ones)*100

    if verbose:
        print('Precision for TEST set _shifted:', Precision_test_shifted)
        print('Sensitivity for TEST set _shifted:', Sensitivity_test_shifted)
        print('Specificity for TEST set _shifted:', Specificity_test_shifted)
        print('F1 Score for TEST set _shifted:', F1_Score_test_shifted)
        print('FPR for TEST set _shifted:', FPR_test_shifted)
        print('very major error for TEST set _shifted:', very_major_error_test_shifted)
        print ('major error for TEST set _shifted: ', major_error_test_shifted)
        print('\n confusion matrix for TEST set _shifted: \n', confusion_test_shifted)

    test_set_metrics_shifted = {"Precision_shifted": Precision_test_shifted, "Sensitivity_shifted": Sensitivity_test_shifted, "Specificity_shifted": Specificity_test_shifted, "F1_Score_shifted": F1_Score_test_shifted, "FPR_shifted": FPR_test_shifted, "VME_shifted": very_major_error_test_shifted, "ME_shifted": major_error_test_shifted, "Confusion_matrix_shifted": confusion_test_shifted}

    #Plot confusion matrix for shifted_thres for TEST set - help from https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
    if output_plots:
        plt.figure()
        names = ['True R','False S','False R','True S']
        counts = ["{0:0.0f}".format(value) for value in confusion_test_shifted.flatten()]
        percentages = ["{0:.1%}".format(value) for value in confusion_test_shifted.flatten()/np.sum(confusion_test_shifted)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(names,counts,percentages)]
        labels = np.asarray(labels).reshape(2,2)

        sns.heatmap(confusion_test_shifted, annot=labels, fmt="", cmap="PuBu", cbar=False)
        plt.xlabel("Predicted", fontsize = 12) #Must be placed after heatmap generation
        plt.ylabel("True", fontsize = 12)
        plt.title("Confusion matrix for TEST set shifted_threshold")

    return test_set_metrics_shifted

#Generate roc curve using predict_proba
def roc_curve_on_training_set_with_proba(self, opt_estimator, opt_features, verbose=True):
    if verbose:
        print("roc_curve_on_training_set_with_proba is working!")

    fpr, tpr, thresholds_proba = roc_curve(self.y_train_nd[opt_features], opt_estimator.predict_proba(self.X_train_nd[opt_features])[:,1])

    if verbose:
        print("positive predict_proba:", opt_estimator.predict_proba(self.X_train_nd[opt_features])[:,1])
        print("predict_proba thresholds:", thresholds_proba)

    plt.figure()
    plt.plot(fpr, tpr, label='RF')
    close_zero = np.argmin(np.abs(thresholds_proba-0.5))
    plt.plot(fpr[close_zero], tpr[close_zero], '^', c='k', markersize=10, label='proba threshold closest to 0.5', fillstyle='none', mew=2)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='best')
    plt.title("ROC curve for TRAINING SET for "+str(opt_features)+" features and "+str(opt_estimator))

    #Calculate ROC_AUC scores for both training and test sets
    training_roc_auc = roc_auc_score(self.y_train_nd[opt_features], opt_estimator.predict_proba(self.X_train_nd[opt_features])[:,1])
    if verbose:
        print("ROC AUC for TRAINING SET:", training_roc_auc)

    test_roc_auc = roc_auc_score(self.y_test_nd[opt_features], opt_estimator.predict_proba(self.X_test_nd[opt_features])[:,1])
    if verbose:
        print("ROC AUC for TEST SET:", test_roc_auc)

#Plots ROC curve for training set and ROC_AUC for training and test sets using decision function
def roc_curve_on_training_set_with_decision(self, opt_estimator, opt_features, verbose=True):
    if verbose:
        print("roc_curve_on_training_set_with_decision is working!")

    fpr, tpr, thresholds = roc_curve(self.y_train_nd[opt_features], opt_estimator.decision_function(self.X_train_nd[opt_features]))

    if verbose:
        print("decision thresholds:", thresholds)

    plt.figure()
    plt.plot(fpr, tpr, label='LogReg for training set')
    close_zero = np.argmin(np.abs(thresholds))
    plt.plot(fpr[close_zero], tpr[close_zero], '^', c='k', markersize=10, label='threshold zero ', fillstyle='none', mew=2)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='best')
    plt.title("ROC curve for TRAINING SET for "+str(opt_features)+" features and "+str(opt_estimator))

    #Calculate ROC_AUC scores for both training and test sets
    training_roc_auc = roc_auc_score(self.y_train_nd[opt_features], opt_estimator.decision_function(self.X_train_nd[opt_features]))
    if verbose:
        print("ROC AUC for TRAINING SET:", training_roc_auc)

    test_roc_auc = roc_auc_score(self.y_test_nd[opt_features], opt_estimator.decision_function(self.X_test_nd[opt_features]))
    if verbose:
        print("ROC AUC for TEST SET:", test_roc_auc)
