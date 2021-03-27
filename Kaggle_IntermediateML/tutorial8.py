import numpy as np 
import pandas as pd 
import matploblib.pyplot as plt 

from sklearn.model_seelction import train_test_split 
from sklearn.metrics import roc_auc_score 
from sklearn.preprocessing import LabelEncoder 

from sklearn.ensemble import RandomForestClassifier 

input_path = Path('/kaggle/input/tabular-playground-series-mar-2021/')
train = pd.read_csv(input_path / 'train.csv', index_col='id')
display(train.head()) 

test = pd.read_csv(input_path / 'test.csv', index_col='id')
display(test.head())

submission = pd.read_csv(input_path / 'sample_submission.csv', index_col='id')
display(submission.head())

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(train[c].values)
        test[c] = lbl.transform(test[c].values)

display(train.head())

target = train.pop('target')
X_train, X_test, y_train, y_test = train_test_split(train, target, train_size=0.60)

clf = RandomForestClassifier(n_estimators=200, max_depth=7, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1] # This grabs the positive class prediction 
score = roc_auc_score(y_test, y_pred) 
print(f'{score:0.5f}')

plt.figure(figsize=(8, 4))
plt.hist(y_pred[np.where(y_test == 0)], bins=100, alpha=0.75, label='neg class')
plt.hist(y_pred[np.where(y_test == 1)], bins=100, alpha=0.75, label='pos class')
plt.legend()
plt.show() 

clf = RandomForestClassifier(n_estimators=200, max_depth=7, n_jobs=-1)
clf.fit(train, target) 
submission['target'] = clf.predict_proba(test)[:, 1]
submission.to_csv('random_forst.csv')