import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

path = '/Users/jonghyunchoe/Documents/Others/GitHub/DS_Project_2/CSV/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')

x_cols = ['label', 's1'] 
X = train[x_cols].copy()

# 1. Data Preprocessing
# 데이터를 50개씩, 1000개씩 묶어서 처리 

# 1-1. Training Data
sum = 0
X['s1_mean_50'] = 0
for i in X.index:
    sum += X.iloc[i, 1]
    if (i%50 == 49):
        X.iloc[(i-49):(i+1), 2] = sum / 50
        sum = 0
    elif (i == X.index[-1]):
        X.iloc[(i-i%50):(i+1), 2] = sum / 50
        sum = 0

X['s1_mean_1000'] = 0
for i in X.index:
    sum += X.iloc[i, 1]
    if (i%1000 == 999):
        X.iloc[(i-999):(i+1), 3] = sum / 1000
        sum = 0
    elif (i == X.index[-1]):
        X.iloc[(i-i%1000):(i+1), 3] = sum / 1000
        sum = 0

# 1-2. Test Data
X_test = test[['s1']].copy()

sum = 0
X_test['s1_mean_50'] = 0
for i in X_test.index:
    sum += X_test.iloc[i, 0]
    if (i%50 == 49):
        X_test.iloc[(i-49):(i+1), 1] = sum / 50
        sum = 0
    elif (i == X_test.index[-1]):
        X_test.iloc[(i-i%50):(i+1), 1] = sum / 50 
        sum = 0

X_test['s1_mean_1000'] = 0
for i in X_test.index:
    sum += X_test.iloc[i, 0]
    if (i%1000 == 999):
        X_test.iloc[(i-999):(i+1), 2] = sum / 1000
        sum = 0
    elif (i == X_test.index[-1]):
        X_test.iloc[(i-i%1000):(i+1), 2] = sum / 1000
        sum = 0     

print("----------- Preprocessing Complete -----------")

# Save the resulting dataframe as a csv file to save time 
X.to_csv(path + 'window_preprocessed_train.csv') 
X_test.to_csv(path + 'window_preprocessed_test.csv')

X = pd.read_csv(path + 'window_preprocessed_train.csv') 
X_test = pd.read_csv(path + 'window_preprocessed_test.csv')

# 2. Time Dependency
# 좌우로 20개씩, 총 41개의 시간에서 sensor 값을 보고 label 예측
# LinearRegression, CatBoost, MLP

# 2-1. Training Data
import math 

window_df = pd.DataFrame(index=range(math.ceil(X.index[-1]/50)), columns = ['label', 't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29', 't30', 't31', 't32', 't33', 't34', 't35', 't36', 't37', 't38', 't39', 't40'])

for i in range(math.ceil(X.index[-1]/50)):
    for j in range(0, 41):
        count = -1000
        if (50*i + 50*j + count >= 0 and 50*i + 50*j + count <= X.index[-1]):
            window_df.iloc[i, j+1] = X.iloc[50*i + 50*j + count, 3]
        else:
            window_df.iloc[i, j+1] = 1403.75
    window_df.iloc[i, 0] = X.iloc[50*i, 1] 

window_df.to_csv(path + 'window_sensor.csv') 

window_df = pd.read_csv(path + 'window_sensor.csv')

window_X = window_df[['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29', 't30', 't31', 't32', 't33', 't34', 't35', 't36', 't37', 't38', 't39', 't40']].copy()
window_y = window_df[['label']].copy() 

# window_X를 93칸만큼 밀어주기 
window_X = window_X.shift(-93) 
window_X = window_X.fillna(1403.75)
pushed_window_df = pd.concat([window_X, window_y], axis=1)
pushed_window_df.to_csv(path + 'window_sensor_pushed.csv')

from sklearn.linear_model import LinearRegression 
from catboost import CatBoostRegressor

model = LinearRegression()
# model = CatBoostRegressor(random_seed = 0, loss_function = 'MAE') #, iterations = iter)

model.fit(window_X, window_y) 
# window_X[['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29', 't30', 't31', 't32', 't33', 't34', 't35', 't36', 't37', 't38', 't39', 't40']] = window_X[['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29', 't30', 't31', 't32', 't33', 't34', 't35', 't36', 't37', 't38', 't39', 't40']].astype(int) 
# window_y[['label']] = window_y[['label']].astype(int) 
# model.fit(window_X, window_y, verbose = False)

# 2-2. Test Data

test_window_df = pd.DataFrame(index=range(math.ceil(X_test.index[-1]/50)), columns = ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29', 't30', 't31', 't32', 't33', 't34', 't35', 't36', 't37', 't38', 't39', 't40'])

for i in range(math.ceil(X_test.index[-1]/50)):
    for j in range(0, 41):
        count = -1000
        if (50*i + 50*j + count >= 0 and 50*i + 50*j + count <= X_test.index[-1]):
            test_window_df.iloc[i, j] = X_test.iloc[50*i + 50*j + count, 2]
        else:
            test_window_df.iloc[i, j] = 1403.75

test_window_df.to_csv(path + 'test_window_sensor.csv')

test_window_df = pd.read_csv(path + 'test_window_sensor.csv')

test_window_X = test_window_df[['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29', 't30', 't31', 't32', 't33', 't34', 't35', 't36', 't37', 't38', 't39', 't40']].copy()

# test_window_X를 93 * 50 칸만큼 밀어주기 
test_window_X = test_window_X.shift(-93) 
test_window_X = test_window_X.fillna(1403.75)
print(test_window_X)

preds = model.predict(test_window_X)

for i in range(len(preds)):
    if (i*50+49 <= submission.index[-1]):
        # 예측값이 100 이하면 0으로 처리 
        if (preds[i][0] <= 100):
            preds[i][0] = 0 
        submission.iloc[(i*50):(i*50+50), 1] = preds[i][0]
    else:
        if (preds[i][0] <= 100):
            preds[i][0] = 0 
        submission.iloc[(i*50):(submission.index[-1]+1), 1] = preds[i][0]

window_df = pd.read_csv(path + 'window_sensor.csv')
window_X = window_df[['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29', 't30', 't31', 't32', 't33', 't34', 't35', 't36', 't37', 't38', 't39', 't40']].copy()
window_X = window_X.shift(-93) 
window_X = window_X.fillna(1403.75)
preds_train = model.predict(window_X)
predicted_train = train[['label', 's1']].copy()
predicted_train['predicted_label'] = 0 

for i in range(len(preds_train)):
    if (i*50+49 <= train.index[-1]):
        # 예측값이 100 이하면 0으로 처리 
        if (preds_train[i][0] <= 100):
            preds_train[i][0] = 0 
        predicted_train.iloc[(i*50):(i*50+50), 2] = preds_train[i][0]
    else:
        if (preds_train[i][0] <= 100):
            preds_train[i][0] = 0 
        predicted_train.iloc[(i*50):(predicted_train.index[-1]+1), 2] = preds_train[i][0]

predicted_train.to_csv(path + 'window_predicted_train.csv', index=False)

predicted_train['predicted_label_mean'] = 0 
sum = 0
for i in predicted_train.index:
    sum += predicted_train.iloc[i, 2] 
    if (i%1000 == 999):
        predicted_train.iloc[(i-999):(i+1), 3] = sum / 1000
        sum = 0
    elif (i == predicted_train.index[-1]):
        predicted_train.iloc[(i-i%1000):(i+1), 3] = sum / 1000
        sum = 0 

predicted_train.to_csv(path + 'window_average_predicted_train.csv', index=False)

predicted_train = pd.read_csv(path + 'window_average_predicted_train.csv')

start = 0
end = 1000
increasing = 1
prev_value = predicted_train.iloc[0, 3]
peak = prev_value
trough = prev_value 
predicted_train['predicted_label_box'] = 0 

for i in range(0, predicted_train.index[-1]+1, 1000):
    next_value = predicted_train.iloc[i, 3]
    # 이전 구간이 올라가는 구간일 때 
    if (increasing): 
        start_value = predicted_train.iloc[start, 3] 
        end_value = predicted_train.iloc[end, 3] 

        # 값이 높은 구간에서 일정 값 이하 변동이 일어날 경우 구간 확장하기 
        if (start_value >= 300 and (abs(start_value - end_value) < 40)):
            end = i + 1000
        # 구간을 끝내고 올라가는 구간 내의 label 값 설정하기 
        elif (next_value < prev_value and (end - start) >= 8000):
            predicted_label = peak 
            print("Rising interval  - start : ", int(start/100), " end: ", int(end/100), " peak: ", peak, " predicted_label: ", predicted_label)
            if (start_value == 0):
                start_index = start + 40*100
            end_index = end 
            predicted_train.iloc[start_index:end_index, 4] = predicted_label 

            increasing = 0
            start = i 
            end = i + 1000 
            trough = next_value 
        # 구간 확장하기 
        else:
            if (prev_value == 0 and next_value == 0):
                start = i 
            peak = next_value 
            end = i + 1000
    
    # 이전 구간이 내려가는 구간일 때 
    else: 
        # 구간을 끝내고 내려가는 구간의 label 값 설정하기 
        if (next_value >= prev_value): # and not (prev_value == 0 and next_value == 0) and (end - start) >= 8000):
            predicted_label = peak # trough
            print("Falling interval - start : ", int(start/100), " end: ", int(end/100), " peak: ", peak, " predicted_label: ", predicted_label)
            start_index = start
            if (next_value == 0 and (end - 1000) > start):
                end_index = end - 1000
            else:
                end_index = end
            predicted_train.iloc[start_index:end_index, 4] = predicted_label 

            increasing = 1 
            start = i 
            end = i + 1000 
            peak = next_value 
        # 구간 확장하기 
        else: 
            # if (prev_value == 0 and next_value == 0):
            #     start = i 
            # trough = next_value 
            end = i + 1000 
    prev_value = next_value 

predicted_train.to_csv(path + 'box_window_predicted_train.csv', index=False)

submission.to_csv(path + 'window_predicted_test.csv', index=False)

submission = pd.read_csv(path + 'window_predicted_test.csv')

sum = 0
for i in submission.index:
    sum += submission.iloc[i, 1] 
    if (i%1000 == 999):
        submission.iloc[(i-999):(i+1), 1] = sum / 1000
        sum = 0
    elif (i == submission.index[-1]):
        submission.iloc[(i-i%1000):(i+1), 1] = sum / 1000
        sum = 0 

submission.to_csv(path + 'window_average_predicted_test.csv', index=False)

# 3. Box Prediction
# 구간별 알고리즘 적용 
# 위 예측값으로 구간 통일 
# start, end, height 예측 

predicted_test = pd.read_csv(path + 'window_average_predicted_test.csv')

start = 0 
end = 1000 
increasing = 1
prev_value = predicted_test.iloc[0, 1]
peak = prev_value 
trough = prev_value 
predicted_test['predicted_label_box'] = 0 

for i in range(0, predicted_test.index[-1]+1, 1000):
    next_value = predicted_test.iloc[i, 1]
    # 이전 구간이 올라가는 구간일 경우 
    if (increasing):
        start_value = predicted_test.iloc[start, 1] 
        end_value = predicted_test.iloc[end, 1] 

        # 값이 높은 구간에서 일정 값 이하 변동이 일어날 경우 구간 확장하기 
        if (start_value >= 300 and (abs(start_value - end_value) < 40)):
            end = i + 1000
        # s1_mean이 내려가면 올라가는 구간을 끝내고 내려가는 구간 시작하기
        if (next_value < prev_value and (end - start) >= 8000):
            # 올라가는 구간 내의 label 값 설정하기
            predicted_label = peak
            print("Rising interval  - start : ", int(start/100), " end: ", int(end/100), " peak: ", peak, " predicted_label: ", predicted_label)
            if (start_value == 0):
                start_index = start + 40*100
            end_index = end 
            predicted_test.iloc[start_index:end_index, 2] = predicted_label 

            increasing = 0
            start = i 
            end = i + 1000 
            trough = next_value 
        # s1_mean이 올라가면 end를 업데이트해 구간 확장하기
        else:
            if (prev_value == 0 and next_value == 0):
                start = i 
            peak = next_value 
            end = i + 1000

    # 이전 구간이 내려가는 구간일 경우 
    else:
        # s1_mean이 올라가면 내려가는 구간을 끝내고 올라가는 구간 시작하기
        if (next_value >= prev_value):
            # 내려가는 구간 내의 label 값 설정하기 
            predicted_label = peak 
            print("Falling interval - start : ", int(start/100), " end: ", int(end/100), " trough: ", trough, " predicted_label: ", predicted_label)
            start_index = start
            if (next_value == 0 and (end - 1000) > start):
                end_index = end - 1000
            else:
                end_index = end
            predicted_test.iloc[start_index:end_index, 2] = predicted_label 
            
            increasing = 1 
            start = i 
            end = i + 1000 
            peak = next_value 
        # s1_mean이 내려가면 end를 업데이트해 구간 확장하기 
        else:
            end = i + 1000

    prev_value = next_value 

predicted_test.to_csv(path + 'visualization_box_window_predicted_test.csv', index=False)
submission = predicted_test[['id', 'predicted_label_box']].copy()
submission = submission.rename(columns = {'predicted_label_box': 'label'}, inplace=False)
submission.to_csv(path + 'box_window_predicted_test.csv', index=False)
