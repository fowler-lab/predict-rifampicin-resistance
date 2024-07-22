# Ml-models
Ml-models is a Python package for running various machine learning models and optimising their hyperparameters.

To run: 
```
Models({"dataframe_name": dataframe}, "model", "CV_scoring_metric", random_seed=(int), test_fraction=(int)).returning_output()
```

- `{"dataset_name": dataframe}` - A dictionary of the input data, with `key` being a user-chosen name ("name") for the dataframe and the `value` being a Pandas dataframe, `dataframe`, for the input dataset, with the last column named "BF" for the binary phenotype. If multiple datasets are provided, the code will optimise the hyperparameters for each dataset in turn and then output metrics for the dataset with the highest CV score. 
- "model" - This is the classifier e.g. logistic regression, linear SVM, random forest, decision tree or gradient-boosted decision tree. Options: "LR", "LSVM", "RF", "DT", "GBDT" or "ALL" (where the "ALL" runs the LR, RF, DT and GBDT classifiers.)
- "CV_scoring_metric" - This is the scoring metric for the cross-validation score. Options: "accuracy", "average_precision", "roc_auc".
- `random_seed=` - The random state for splitting the dataset into training and test sets. Use the same value for reproducible splitting.  Options: integer. 
- `test_fraction=` - The fraction with which to split the dataset into test and training sets. Options: float between 0 and 1, where 0 is no test data and 1 is all test data. 
- `.returning_output()` - Provides the full output for training and running the models. The metrics can subsequently be saved using skops. 
- `get_model()` - Outputs the final model, which can subsequently be saved using skops.
- `get_threshold()` - Outputs the decision threshold for maximising the F1 score, which is used to calculate the "shifted_threshold" output metrics. This can subsequently be saved using skops.