import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer, normalize, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge
from sklearn.svm import SVR


def comma_to_dot_float(s):
    if type(s) == str:
        return float(s.replace(',', '.'))
    return s


def preprocess(df, train=True):
    # convert comma strings to floats
    cols = ['pa_lrn_supp_home', 'pa_emotional_supp', 'fam_wealth',
            'escs_index', 'age', 'st_lrn_min_math', 'st_lrn_min_lang',
            'st_lrn_min_sci', 'st_lrn_min_total', 'st_home_edu_resources',
            'st_home_poss', 'st_life_satis', 'sch_stu_teacher_ratio',
            'sch_num_teachers', 'sch_comp_ratio', 'sch_comp_int_ratio']
    if train:
        cols.append('math')
    df.loc[:, cols] = df.loc[:, cols].applymap(comma_to_dot_float)

    # TODO handle missing values (remove, input median, etc)

    # HANDLE CATEGORICAL DATA
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
    df.st_got_hit = df.st_got_hit.map({'Never or almost never': 0, 'A few times a year': 1,
                                       'A few times a month': 2, 'Once a week or more': 3})
    df.st_took_away_things = df.st_took_away_things.map({'Never or almost never': 0, 'A few times a year': 1,
                                                         'A few times a month': 2, 'Once a week or more': 3})
    df.st_made_fun = df.st_made_fun.map({'Never or almost never': 0, 'A few times a year': 1,
                                         'A few times a month': 2, 'Once a week or more': 3})
    df.st_threatened = df.st_threatened.map({'Never or almost never': 0, 'A few times a year': 1,
                                             'A few times a month': 2, 'Once a week or more': 3})

    # combinacion de les desgracies XD
    df['bullied'] = df.st_rumours + df.st_got_hit + df.st_took_away_things + df.st_made_fun + df.st_threatened

    df.highest_edu_pa = df.highest_edu_pa.map({'None': 0, 'ISCED 1': 1, 'ISCED 2': 2,
                                               'ISCED 3B, C': 3, 'ISCED 3A, ISCED 4': 4,
                                               'ISCED 5B': 5, 'ISCED 5A, 6': 6})

    # generate dummies
    columns_to_dummies = ['pa_paid_edu', 'pa_annual_income', 'st_mother_sch_level', 'st_father_sch_level', 'sch_type']
    df = pd.get_dummies(df, columns=columns_to_dummies)
    return df


# load the data
df_train = pd.read_csv('train_student.csv', sep=';')
df_test = pd.read_csv('test_student.csv', sep=';')
schools = pd.read_csv('school_dataset.csv', sep=';')
df_train = df_train.merge(schools, on = 'sch_id')
df_test = df_test.merge(schools, on = 'sch_id')

df_train = df_train.fillna(0)
df_test = df_test.fillna(0)

df_train_clean = preprocess(df_train)
df_test_clean = preprocess(df_test, train=False)

# choose the model
# clf = RandomForestRegressor(n_estimators=100)
# clf = LinearRegression()
clf = BayesianRidge()
# clf = SVR()

# columns to train
features = df_train_clean.dtypes[df_train_clean.dtypes != 'object'].index.values
features = features[~pd.Series(features).isin(['st_id', 'math', 'sch_id'])]

important_features = ['st_grade', 'st_lrn_min_sci', 'escs_index', 'st_home_poss',
       'fam_wealth', 'pa_lrn_supp_home', 'st_lrn_min_total', 'st_gender',
       'sch_stu_teacher_ratio', 'sch_comp_ratio']


# target column
y = df_train_clean.math

# imput missing values
X, X_test = df_train_clean.loc[:, features].values, df_test_clean.loc[:, features].values
imp = Imputer(strategy='most_frequent')
X, X_test = imp.fit_transform(X), imp.fit_transform(X_test)

# X = normalize(X, axis=0)
# X_test = normalize(X_test, axis=0)

# train and score the model
# TODO provar diferents tipus de models: lineal, XGBoost, RF
scores = cross_val_score(clf, X, y, cv=10, scoring='neg_mean_squared_error')
print(scores.mean())

# TODO els scores al predict em queden com a maxim a 7.5. No arriben a prop del 10.
# TODO Normalitzo math? Normalitzo totes les variables en general?

# Predict and write
"""
clf.fit(X, y)
y_test = clf.predict(X_test)
submission = pd.DataFrame({'st_id': df_test_clean.st_id, 'score': y_test})
submission = submission[['st_id', 'score']]
submission.score = submission.score.clip(lower=2.5)
submission.to_csv('output/model_linear_improved_normalized.csv', sep=',', index=False)
"""