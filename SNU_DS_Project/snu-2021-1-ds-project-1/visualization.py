import pandas as pd 
from sklearn.model_selection import train_test_split, KFold, GridSearchCV  
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso   
from lightgbm import LGBMRegressor 
from pycaret.regression import *
import matplotlib.pyplot as plt

# 데이터를 읽어들입니다.
X = pd.read_csv('/Users/jonghyunchoe/Documents/College/2021_Spring_Semester/창의융합프로젝트/snu-2021-1-ds-project-1/wine_train.csv')
X_test_full = pd.read_csv('/Users/jonghyunchoe/Documents/College/2021_Spring_Semester/창의융합프로젝트/snu-2021-1-ds-project-1/wine_test.csv')

# predictor variable과 target variable을 분리합니다.
y = X['points']
X.drop(['points'], axis=1, inplace=True)

X = X.iloc[:100]
y = y.iloc[:100]

# training data에서 validation set을 나눠줍니다.
from sklearn.model_selection import train_test_split

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# 이 예시에서는 price, province, taster_name, variety를 predictor variable로 사용했습니다. 이 variable들이 유용할지는 알지 못합니다.
feature_columns = ['id'] # , 'price']

X_train = X_train_full[feature_columns].copy()
X_valid = X_valid_full[feature_columns].copy()
X_test = X_test_full[feature_columns].copy()

from sklearn.preprocessing import OneHotEncoder

categorical_columns = [] # 'province', 'taster_name'

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[categorical_columns]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[categorical_columns]))

# 새로운 DataFrame이 만들어지며 기존 index를 없애버렸으므로 index를 다시 만들어줍니다.
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

from sklearn.impute import SimpleImputer

num_X_train = X_train.drop(categorical_columns, axis=1)
num_X_valid = X_valid.drop(categorical_columns, axis=1)

my_imputer = SimpleImputer(strategy='mean')
imputed_num_X_train = pd.DataFrame(my_imputer.fit_transform(num_X_train))
imputed_num_X_valid = pd.DataFrame(my_imputer.transform(num_X_valid))

# 새로운 DataFrame이 만들어지며 기존 index를 없애버렸으므로 index를 다시 만들어줍니다.
imputed_num_X_train.index = num_X_train.index
imputed_num_X_valid.index = num_X_valid.index

# preprocessing이 완료되었으므로 두 DataFrame을 합쳐줍니다.
preprocessed_X_train = pd.concat([imputed_num_X_train, OH_cols_train], axis=1)
preprocessed_X_valid = pd.concat([imputed_num_X_valid, OH_cols_valid], axis=1)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

model = KNeighborsRegressor(n_neighbors=1) # , weights='distance') 
model.fit(preprocessed_X_train, y_train)
valid_preds = model.predict(preprocessed_X_valid)
print('RMSE: {}'.format(mean_squared_error(y_valid, valid_preds, squared=False)))

# test data에 대해서도 같은 preprocessing 작업을 해준 후, 훈련된 모델을 이용해 결과를 예측합니다.
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[categorical_columns]))
OH_cols_test.index = X_test.index
num_X_test = X_test.drop(categorical_columns, axis=1)
imputed_num_X_test = pd.DataFrame(my_imputer.transform(num_X_test))
imputed_num_X_test.index = num_X_test.index

preprocessed_X_test = pd.concat([imputed_num_X_test, OH_cols_test], axis=1)
test_preds = model.predict(preprocessed_X_test)

import numpy as np
_, axes = plt.subplots(1, 3)

line = np.linspace(0, 100, num=1000)
line = line.reshape(-1, 1)

for i, ax in zip([1, 3, 5], axes.ravel()):
    knn_reg = KNeighborsRegressor(n_neighbors=i, n_jobs=-1)
    knn_reg.fit(preprocessed_X_train, y_train)

    prediction = knn_reg.predict(line)
    ax.plot(line, prediction, label='model predict', c='k') 
    ax.scatter(preprocessed_X_train, y_train, marker='^', c='darkred', label='train target')
    ax.scatter(preprocessed_X_valid, y_valid, marker='v', c='darkblue', label='test target')
    
    train_score = knn_reg.score(preprocessed_X_train, y_train)
    test_score = knn_reg.score(preprocessed_X_valid, y_valid)
    ax.set_title('k={}\ntest score={:.3f}\ntrain score={:.3f}'.format(i, train_score, test_score))
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')
axes[0].legend(loc=2)
plt.show()

# Submission 파일을 만듭니다.
my_submission = pd.DataFrame({'id': preprocessed_X_test.index, 'points': test_preds})
my_submission.to_csv('wine_my_submission.csv', index=False)