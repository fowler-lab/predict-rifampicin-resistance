import numpy as np
import pandas as pd
import sklearn
import pytest
import mlmodels

def test_reading_in_data():
    all_features = pd.read_csv("tests/all_features.csv")
    reduced_features = pd.read_csv("tests/reduced_features.csv")
    total_dict = {"all": all_features, "reduced": reduced_features}

    #Check that data is in dictionary form
    assert isinstance(total_dict, dict)

    #Check that key:values pairs in dictionary are str:dataframes
    for key, df in total_dict.items():
        assert isinstance(key, str)
        assert isinstance(df, pd.DataFrame)

    #Check size of dataframes for each key
    assert np.shape(total_dict["all"]) == (307, 25)
    assert np.shape(total_dict["reduced"]) == (307, 11)

    #tests to check all assert isinstances work correctly
    
