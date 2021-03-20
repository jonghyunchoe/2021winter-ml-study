# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory 
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory 

import os 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames: 
        print(os.path.join(dirname, filename)) 

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# 데이터를 읽어들입니다.
X = pd.read_csv('../input/snu-2021-1-ds-project-1/wine_train.csv', index_col='id')
X_test_full = pd.read_csv('../input/snu-2021-1-ds-project-1/wine_test.csv', index_col='id')

# training data의 형태는 다음과 같습니다.
X.head(10) 

X.tail(10) 

# predictor variable과 target variable을 분리합니다.
y = X['points']
X.drop(['points'], axis=1, inplace=True)

# training data에서 validation set을 나눠줍니다.
from sklearn.model_selection import train_test_split 

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# 전체 data의 column 목록은 다음과 같습니다. 무엇이 point 예측에 유용할지 고민해 보시기 바랍니다.
X_train_full.columns 

# 이 예시에서는 price, province, taster_name, variety를 predictor variable로 사용했습니다. 이 variable들이 유용할지는 알지 못합니다.
feature_columns = ['price', 'province', 'taster_name', 'variety']

X_train = X_train_full[feature_columns].copy() 
X_valid = X_valid_full[feature_columns].copy() 
X_test = X_test_full[feature_columns].copy() 

# province, province, taster_name은 categorical variable입니다. 여기서는 one-hot encoding을 사용하여 preprocessing해 주었습니다.
from sklearn.preprocessing import OneHotEncoder 

categorical_columns = ['province', 'taster_name', 'variety']

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[categorical_columns]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[categorical_columns]))

# 새로운 DataFrame이 만들어지며 기존 index를 없애버렸으므로 index를 다시 만들어줍니다. 이 과정은 필수적이진 않습니다.
OH_cols_train.columns = OH_encoder.get_feature_names(['province', 'taster_name', 'variety'])
OH_cols_valid.columns = OH_encoder.get_feature_names(['province', 'taster_name', 'variety'])

# numerical variable(여기서는 price)의 빠진 값을 적당한 값으로 채워줍니다. 여기서는 training set에서 평균값으로 빠진 값을 채웁니다. 
from sklearn.impute import SimpleImputer 

num_X_train = X_train.drop(categorical_columns, axis=1)
num_X_valid = X_valid.drop(categorical_columns, axis=1)

my_imputer = SimpleImputer(strategy='mean')
imputed_num_X_train = pd.DataFrame(my_imputer.fit_transform(num_X_train))
imputed_num_X_valid = pd.DataFrame(my_imputer.transform(num_X_valid))

# 새로운 DataFrame이 만들어지며 기존 index를 없애버렸으므로 index를 다시 만들어줍니다.
imputed_num_X_train.index = num_X_train.index 
imputed_num_X_valid.index = num_X_valid.index 

# 새로운 DataFrame이 만들어지며 기존 column names를 없애버렸으므로 column names를 다시 만들어줍니다. 이 과정은 필수적이진 않습니다.
imputed_num_X_train.columns = num_X_train.columns 
imputed_num_X_valid.columns = num_X_valid.columns 

# preprocessing이 완료되었으므로 두 DataFrame을 합쳐줍니다.
preprocessed_X_train = pd.concat([imputed_num_X_train, OH_cols_train], axis=1)
preprocessed_X_valid = pd.concat([imputed_num_X_valid, OH_cols_valid], axis=1)

# 모델을 만들고 training 시킵니다. 여기서는 DecisionTreeRegressor를 사용했습니다. 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error

model = DecisionTreeRegressor(random_state=1)
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

# Submission 파일을 만듭니다.
my_submission = pd.DataFrame({'id': preprocessed_X_test.index, 'points': test_preds})
my_submission.to_csv('wine_my_submission.csv', index=False)


