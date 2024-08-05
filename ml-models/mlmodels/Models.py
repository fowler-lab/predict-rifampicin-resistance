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
)

from sklearn.utils import resample

from . import LR
from . import LSVM
from . import DT
from . import GBDT
from . import RF

from . import Analysis


class Models(object):
    """Searches through hyperparameters for different machine learning models."""

    def __init__(
        self,
        input_df_dict,
        model_type,
        scoring_metric,
        random_seed=0,
        test_fraction=0.25,
        upsample=False,
        upsample_coeff=1,
        sample_ids=None,
        verbose=True,
        output_plots=True,
    ):
        assert isinstance(
            input_df_dict, dict
        ), "Input data must be a dictionary with key:value pairs key,int value,pd.DataFrame"
        for key, df in input_df_dict.items():
            assert isinstance(key, str), "Keys must be strings!"
            assert isinstance(df, pd.DataFrame), "Values must be pd.DataFrames!"

        assert isinstance(
            model_type, str
        ), "Machine learning model type must be entered as a string!"
        assert isinstance(
            scoring_metric, str
        ), "Hyperparameter tuning metric must be a string!"
        assert isinstance(random_seed, int), "Random seed must be an integer!"
        assert isinstance(
            upsample, bool
        ), "Upsampling currently not implemented. Once implemented, upsample must be True or False"
        assert isinstance(
            upsample_coeff, int
        ), "Upsampling currently not implemented. Once implemented, coefficient must be an integer!"

        self.input_df = input_df_dict
        self.model = model_type  # can be LR, LSVM, RF, DT, GBDT, all
        self.scoring_metric = scoring_metric  # can be accuracy, precision, roc_auc
        self.random_seed = random_seed
        self.upsample = upsample
        self.upsample_coeff = upsample_coeff
        self.test_fraction = test_fraction
        self.verbose = verbose
        self.output_plots = output_plots
        self.sample_ids = sample_ids

        self.PrepareInputData()

        if model_type == "LR":
            LR.LogisticReg(self, verbose=verbose, output_plots=output_plots)
        elif model_type == "LSVM":
            LSVM.LinearSVM(self, verbose=verbose, output_plots=output_plots)
        elif model_type == "RF":
            RF.RandomForest(self, verbose=verbose, output_plots=output_plots)
        elif model_type == "DT":
            DT.DecisionTree(self, verbose=verbose, output_plots=output_plots)
        elif model_type == "GBDT":
            GBDT.GradientBoostedDecisionTree(
                self, verbose=verbose, output_plots=output_plots
            )
        elif model_type == "ALL":
            LR.LogisticReg(self, verbose=verbose, output_plots=output_plots)
            RF.RandomForest(self, verbose=verbose, output_plots=output_plots)
            DT.DecisionTree(self, verbose=verbose, output_plots=output_plots)
            GBDT.GradientBoostedDecisionTree(
                self, verbose=verbose, output_plots=output_plots
            )
        else:
            raise NameError(
                "This is not a supported model: try LR, LSVM, RF, DT, GBDT or ALL"
            )

        scoring_metric_list = ["accuracy", "average_precision", "roc_auc", "recall"]
        if scoring_metric not in scoring_metric_list:
            raise NameError(
                "This not a supported scoring metric: try accuracy, average_precision, recall, or roc_auc"
            )

    def returning_output(self, output_plots=True):
        if self.scoring_metric == "average_precision":
            output_dict = Analysis.metrics_and_confusion_on_test_set(
                self,
                self.opt_estimator,
                self.opt_features,
                verbose=self.verbose,
                output_plots=output_plots,
            )
            output_dict.update(
                Analysis.metrics_on_test_set_shiftedthres(
                    self,
                    self.opt_estimator,
                    self.opt_features,
                    self.opt_threshold,
                    verbose=self.verbose,
                    output_plots=output_plots,
                )
            )
        elif self.scoring_metric == "recall":
            output_dict = Analysis.metrics_and_confusion_on_test_set(
                self,
                self.opt_estimator,
                self.opt_features,
                verbose=self.verbose,
                output_plots=output_plots,
            )
            output_dict.update(
                Analysis.metrics_on_test_set_shiftedthres(
                    self,
                    self.opt_estimator,
                    self.opt_features,
                    self.opt_threshold,
                    verbose=self.verbose,
                    output_plots=output_plots,
                )
            )
        else:
            output_dict = Analysis.metrics_and_confusion_on_test_set(
                self,
                self.opt_estimator,
                self.opt_features,
                verbose=self.verbose,
                output_plots=output_plots,
            )
        return output_dict

    def get_model(self):
        model = self.opt_estimator
        return model

    def get_threshold(self):
        if self.scoring_metric == "average_precision":
            threshold = self.opt_threshold
        else:
            if self.verbose:
                print("No threshold for accuracy or roc_auc!")
            threshold = None
        return threshold

    def PrepareInputData(self):
        """
        Prepares the training and test datasets with split specified by user, and applies upsampling to the training dataset
        """

        features_array_df = {}
        features_nd_array = {}
        for key, df in self.input_df.items():
            features_array_df[key] = df[df.columns[:-1]]
            features_nd_array[key] = features_array_df[key].to_numpy()
            if self.verbose:
                print(
                    "features_nd_array shape for",
                    [key],
                    np.shape(features_nd_array[key]),
                )

        bf_array_df = {}
        bf_nd_array = {}
        for key, df in self.input_df.items():
            bf_array_df[key] = df[df.columns[-1]]
            bf_nd_array[key] = bf_array_df[key].to_numpy()
            if self.verbose:
                print("bf_nd_array shape for", [key], np.shape(bf_nd_array[key]))
        if self.verbose:
            print("bf_nd_array:", bf_nd_array)

        X_train_nd = {}
        X_test_nd = {}
        y_train_nd = {}
        y_test_nd = {}
        train_ids = {}
        test_ids = {}
        for key, df in self.input_df.items():
            if self.sample_ids is not None:
                X_train, X_test, y_train, y_test, train_id, test_id = train_test_split(
                    features_nd_array[key],
                    bf_nd_array[key],
                    self.sample_ids[key],
                    test_size=self.test_fraction,
                    random_state=self.random_seed,
                )
                train_ids[key] = train_id
                test_ids[key] = test_id
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    features_nd_array[key],
                    bf_nd_array[key],
                    test_size=self.test_fraction,
                    random_state=self.random_seed,
                )

            if self.verbose:
                print("For", key, ": X_test_nd shape =", np.shape(X_test))

            # Apply upsampling if required
            if self.upsample_coeff > 1:
                X_train, y_train = self.upsample_training_data(X_train, y_train)

            X_train_nd[key] = X_train
            X_test_nd[key] = X_test
            y_train_nd[key] = y_train
            y_test_nd[key] = y_test

        self.X_train_nd = X_train_nd
        self.X_test_nd = X_test_nd
        self.y_train_nd = y_train_nd
        self.y_test_nd = y_test_nd
        if self.sample_ids is not None:
            self.train_ids = train_ids
            self.test_ids = test_ids

    def upsample_training_data(self, X_train, y_train):
        """
        Upsamples the positive class in the training data.

        Parameters:
        ----------
        X_train : ndarray
            Training feature data.
        y_train : ndarray
            Training labels.

        Returns:
        -------
        X_train_upsampled : ndarray
            Upsampled training feature data.
        y_train_upsampled : ndarray
            Upsampled training labels.
        """
        # Convert to DataFrame for easier manipulation
        X_train_df = pd.DataFrame(X_train)
        y_train_df = pd.Series(y_train)

        # Separate the positive and negative classes
        X_train_positive = X_train_df[y_train_df == 1]
        y_train_positive = y_train_df[y_train_df == 1]
        X_train_negative = X_train_df[y_train_df == 0]
        y_train_negative = y_train_df[y_train_df == 0]

        # Upsample the positive class
        X_train_positive_upsampled = pd.concat([X_train_positive] * self.upsample_coeff)
        y_train_positive_upsampled = pd.concat([y_train_positive] * self.upsample_coeff)

        # Combine the upsampled positive class with the original negative class
        X_train_upsampled = pd.concat([X_train_negative, X_train_positive_upsampled])
        y_train_upsampled = pd.concat([y_train_negative, y_train_positive_upsampled])

        # Shuffle the upsampled training data
        X_train_upsampled, y_train_upsampled = sklearn.utils.shuffle(
            X_train_upsampled, y_train_upsampled, random_state=self.random_seed
        )

        return X_train_upsampled.to_numpy(), y_train_upsampled.to_numpy()
