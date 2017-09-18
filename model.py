import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer


def comma_to_dot_float(s):
    if type(s) == str:
        return float(s.replace(',', '.'))
    return s


def preprocess(df, train=True):
    # convert comma strings to floats
    cols = ['pa_lrn_supp_home', 'pa_emotional_supp', 'fam_wealth',
            'escs_index', 'age', 'st_lrn_min_math', 'st_lrn_min_lang',
            'st_lrn_min_sci', 'st_lrn_min_total', 'st_home_edu_resources',
            'st_home_poss', 'st_life_satis']
    if train:
        cols.append('math')
    df.loc[:, cols] = df.loc[:, cols].applymap(comma_to_dot_float)

    # TODO handle missing values (remove, input median, etc)

    # handle categorical data
    categorical_columns = df.dtypes[df.dtypes == object].index.values

    return df


# load the data
df_train = pd.read_csv('train_student.csv', sep=';')
df_test = pd.read_csv('test_student.csv', sep=';')

df_train_clean = preprocess(df_train)
df_test_clean = preprocess(df_test, train=False)

# choose the model
clf = RandomForestRegressor(n_estimators=100)

# columns to train
features = df_train_clean.dtypes[df_train_clean.dtypes != 'object'].index.values
features = features[features != 'math']
# target column
y = df_train_clean.math

# imput missing values
X = df_train_clean.loc[:, features].values
imp = Imputer()
X = imp.fit_transform(X)

# train and score the model
scores = cross_val_score(clf, X, y, cv=10, scoring='neg_mean_squared_error')