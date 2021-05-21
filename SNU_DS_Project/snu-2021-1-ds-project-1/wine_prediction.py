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

X_full = pd.read_csv('/Users/jonghyunchoe/Documents/College/2021_Spring_Semester/창의융합프로젝트/snu-2021-1-ds-project-1/wine_train.csv')
X_test_full = pd.read_csv('/Users/jonghyunchoe/Documents/College/2021_Spring_Semester/창의융합프로젝트/snu-2021-1-ds-project-1/wine_test.csv')

y = X_full.points

X_full.drop(['points'], axis=1, inplace=True)
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)

categorical_cols = ['country', 'province', 'region_1', 'region_2'] # ['country'] # , 'variety', 'region_2' 'province', 'region_1', 'region_2', 'taster_name'
# categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 1000 and 
#                    X_train_full[cname].dtype == "object"]
numerical_cols = ['id', 'year'] # [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
print("Used columns:", my_cols)
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# Step 1: Define Preprocessing Steps 
numerical_transformer = SimpleImputer(strategy='median') # strategy='mean', 'median', 'most_frequent', 'constant'
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # strategy='mean', 'median', 'most_frequent', 'constant'
    ('onehot', OneHotEncoder(handle_unknown='ignore')) 
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

print("----------- Step 1 Complete -----------") 

# Step 2: Define the Model 
# model = DecisionTreeRegressor(random_state=0) 
# model = RandomForestRegressor(n_estimators=10, random_state=0)
# model = LGBMRegressor(n_estimators=100000, learning_rate=0.01) # num_leaves, max_depth, learning_rate, n_estimators
# model = GradientBoostingRegressor() 
# model = AdaBoostRegressor() 
# model = Ridge() 
# model = Lasso() 
# model = LinearRegression() 
model = KNeighborsRegressor(n_neighbors=4, weights='distance') 

print("----------- Step 2 Complete -----------") 

# Step 3: Create and Evaluate the Pipeline 
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])
my_pipeline.fit(X_train, y_train) 

kfold = KFold(n_splits=5, shuffle=True, random_state=0)
params = {
    'model__n_neighbors': [2, 3, 4, 5, 6],
    'model__weights': ['uniform', 'distance']
}

grid = GridSearchCV(estimator=my_pipeline, param_grid=params, cv=kfold, scoring='neg_root_mean_squared_error')
grid.fit(X_train, y_train) 

my_pipeline = grid.best_estimator_
preds = my_pipeline.predict(X_valid) 
score = mean_absolute_error(y_valid, preds)

print("----------- Step 3 Complete -----------") 
print('MAE:', score) 

X_test = X_test_full[my_cols].copy()
predictions = my_pipeline.predict(X_test) 
output = pd.DataFrame({'id': X_test_full.id, 'points': predictions})
output.to_csv('my_submission.csv', index=False)

# Step 4: Model Comparison Through Pycaret 
# reg = setup(data = X_full, 
#              target = 'points',
#              numeric_imputation = 'median',
#              categorical_features = ['country', 'province', 'region_1', 'region_2', 'taster_name', 'variety', 'winery'], 
#              ignore_features = ['description', 'designation', 'taster_twitter_handle', 'title'],
#              normalize = True,
#              silent = True)

# Returns the best model 
# best = compare_models() 