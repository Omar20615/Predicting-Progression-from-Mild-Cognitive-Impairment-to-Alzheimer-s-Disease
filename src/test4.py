import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('new_output_data.csv')
print("Dataset head:")
print(data.head())
scaler = StandardScaler()
nm_columns = data.columns[4:12]
biomarker_columns = data.columns[12:19]
data[nm_columns] = data[nm_columns]
data[biomarker_columns] = data[biomarker_columns]
viscodes = [1,2,3,4,5,6]
data = data[data['VISCODE'].isin(viscodes)]


data_spm = data
data_mptm = data



def predict_future_values_spm(data, target_columns, target_viscode):
    predictions = []
    count = 0
    for column in target_columns:
        train_data = data[(data['VISCODE'] < target_viscode)]
        test_data = data[data['VISCODE'] == target_viscode]
        count +=1
        if train_data.empty or test_data.empty:
            print(f"No data available for training or testing for VISCODE {target_viscode} for column {column}")
            continue
        
        X_train = train_data[[column]]
        y_train = train_data[column]
        
        X_test = test_data[[column]]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        predictions.append((column, test_data['RID'], pred, test_data[column]))
    
    return predictions


pred_24_months_spm = predict_future_values_spm(data_spm, nm_columns.union(biomarker_columns), 5)
for column, rids, preds, actuals in pred_24_months_spm:
    data_spm.loc[data_spm['VISCODE'] == 5, column] = preds

pred_30_months_spm = predict_future_values_spm(data_spm, nm_columns.union(biomarker_columns), 6)
for column, rids, preds, actuals in pred_30_months_spm:
    data_spm.loc[data_spm['VISCODE'] == 6, column] = preds




def predict_future_values_mptm(data, target_columns, target_viscode):
    count = 0
    predictions = []

    test_data = data[data['VISCODE'] == target_viscode]
    count += 1
    X_test = test_data.drop(['Serial Number', 'RID', 'VISCODE', 'Labels'], axis=1)

    preds = []
    for t in viscodes:
        if t == target_viscode:
            break

        train_data = data[data['VISCODE'] == t]
        X_train = train_data.drop(['Serial Number', 'RID', 'VISCODE', 'Labels'], axis=1)
        y_train = train_data.drop(['Serial Number', 'RID', 'VISCODE', 'Labels'], axis=1)

        if X_train.empty or y_train.empty:
            print(f"No samples available for training at VISCODE {t} for column {column}")
            continue
            
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        preds.append(pred)
        
    pred_combined = np.mean(preds, axis=0)
    
    predictions.append((column, test_data['RID'], pred_combined, test_data[column]))
    
    return predictions

pred_24_months_mptm = predict_future_values_mptm(data_mptm, nm_columns.union(biomarker_columns), 5)
for column, rids, preds, actuals in pred_24_months_mptm:
    data_mptm.loc[data_mptm['VISCODE'] == 5, column] = preds

pred_30_months_mptm = predict_future_values_mptm(data_mptm, nm_columns.union(biomarker_columns), 6)

for column, rids, preds, actuals in pred_30_months_mptm:
    data_mptm.loc[data_mptm['VISCODE'] == 6, column] = preds

# def count_close_predictions(predictions, tolerance=0.00001):
#     close_predictions = 0
#     total_predictions = 0

#     for column, rids, preds, actuals in predictions:
#         total_predictions += len(preds)
#         close_predictions += np.sum(np.abs(preds - actuals) <= tolerance)

#     return close_predictions, total_predictions

# close_predictions_24_spm, total_predictions_24_spm = count_close_predictions(pred_24_months_spm)
# close_predictions_30_spm, total_predictions_30_spm = count_close_predictions(pred_30_months_spm)
# print(f"SPM - 24-Month Predictions: {close_predictions_24_spm} out of {total_predictions_24_spm} are close to the actual values.")
# print(f"SPM - 30-Month Predictions: {close_predictions_30_spm} out of {total_predictions_30_spm} are close to the actual values.")
data_spm.to_csv('output_predicted_spm_no_lag.csv', index=False)


#close_predictions_24_mptm, total_predictions_24_mptm = count_close_predictions(pred_24_months_mptm)
#close_predictions_30_mptm, total_predictions_30_mptm = count_close_predictions(pred_30_months_mptm)
#print(f"MPTM - 24-Month Predictions: {close_predictions_24_mptm} out of {total_predictions_24_mptm} are close to the actual values.")
#print(f"MPTM - 30-Month Predictions: {close_predictions_30_mptm} out of {total_predictions_30_mptm} are close to the actual values.")
data_mptm.to_csv('output_predicted_mptm_no_lag.csv', index=False)


original_data = pd.read_csv('new_output_data.csv')
nm_columns = data.columns[4:12]
biomarker_columns = data.columns[12:19]
original_data[nm_columns] = original_data[nm_columns]
original_data[biomarker_columns] = original_data[biomarker_columns]
viscodes = [1,2,3,4,5,6]
original_data = original_data[original_data['VISCODE'].isin(viscodes)]  
are_identical = original_data.equals(data_spm)
assert original_data.shape == data_spm.shape, "Dataframes are of different shapes."

identical_rows = (original_data == data_mptm).all(axis=1)

num_identical_rows = identical_rows.sum()
num_different_rows = len(original_data) - num_identical_rows

print(f"Number of identical rows: {num_identical_rows}")
print(f"Number of different rows: {num_different_rows}")


are_identical = original_data.equals(data_spm)
assert original_data.shape == data_spm.shape, "Dataframes are of different shapes."

identical_rows = (original_data == data_spm).all(axis=1)

num_identical_rows = identical_rows.sum()
num_different_rows = len(original_data) - num_identical_rows

print(f"Number of identical rows: {num_identical_rows}")
print(f"Number of different rows: {num_different_rows}")