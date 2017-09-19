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
    # maps the st_grade columns -> Grade7: 0, Grade 8: 1, etc...
    df.st_grade = df.st_grade.map(lambda x: int(x.split(' ')[1]) - 7)
    # maps male, female to 1, 0
    df.st_gender = df.st_gender.apply(lambda x: 0 if x == 'Male' else 1)
    # maps grade repetition to 1, 0
    df.grade_repetition = df.grade_repetition.str.contains('not').apply(lambda x: 1 if x else 0)
    # pass categorical to ordinal. Careful we are not imputing NANs here
    df.st_rumours = df.st_rumours.map({'Never or almost never': 0, 'A few times a year': 1,
                                   'A few times a month': 2, 'Once a week or more': 3})

    # TODO afegir la variable is_bullied com a combinacio de les desgracies
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
features = features[~pd.Series(features).isin(['st_id', 'math'])]

# target column
y = df_train_clean.math

# imput missing values
X, X_test = df_train_clean.loc[:, features].values, df_test_clean.loc[:, features].values
imp = Imputer()
X, X_test = imp.fit_transform(X), imp.fit_transform(X_test)

# train and score the model
# TODO provar diferents tipus de models: lineal, XGBoost, RF
scores = cross_val_score(clf, X, y, cv=10, scoring='neg_mean_squared_error')
print(scores.mean())

# TODO els scores al predict em queden com a maxim a 7.5. No arriben a prop del 10.
# TODO Normalitzo math? Normalitzo totes les variables en general?

# Predict and write
clf.fit(X, y)
y_test = clf.predict(X_test)
submission = pd.DataFrame({'st_id': df_test_clean.st_id, 'score': y_test})
submission = submission[['st_id', 'score']]
submission.to_csv('model2.csv', sep=',', index=False)