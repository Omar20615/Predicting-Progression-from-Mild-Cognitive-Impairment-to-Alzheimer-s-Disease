import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv('output_complete.csv')
print("Dataset head:")
print(data.head())

nm_columns = data.columns[4:12]
biomarker_columns = data.columns[12:268]
scaler = StandardScaler()
data[nm_columns] = scaler.fit_transform(data[nm_columns])
data[biomarker_columns] = scaler.fit_transform(data[biomarker_columns])

def predict_future_values_spm(data, target_columns, target_viscode):
    predictions = []
    count = 0
    for column in target_columns:
        train_data = data[data['VISCODE'] < target_viscode]
        test_data = data[data['VISCODE'] == target_viscode]
        count +=1
        if train_data.empty or test_data.empty:
            print(f"No data available for training or testing for VISCODE {target_viscode} for column {column}")
            continue
        
        X_train = train_data.drop(['Serial Number', 'RID', 'VISCODE', 'Labels'], axis=1)
        y_train = train_data[column]
        
        X_test = test_data.drop(['Serial Number', 'RID', 'VISCODE', 'Labels'], axis=1)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        predictions.append((column, test_data['RID'], pred, test_data[column]))
    
    return predictions

def predict_future_values_mptm(data, target_columns, target_viscode):
    count = 0

    predictions = []
    for column in target_columns:
        train_data = data[data['VISCODE'] < target_viscode]
        test_data = data[data['VISCODE'] == target_viscode]
        count +=1

        if train_data.empty or test_data.empty:
            print(f"No data available for training or testing for VISCODE {target_viscode} for column {column}")
            continue
        
        # Separate predictions for each time stamp
        preds = []
        for t in range(1, target_viscode - 1):
            X_train = train_data[train_data['VISCODE'] == t][biomarker_columns]
            y_train = train_data[train_data['VISCODE'] == target_viscode][column]
            X_test = test_data[test_data['VISCODE'] == target_viscode][biomarker_columns]
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            preds.append(pred)
        
        # Combine predictions using the formula
        pred_combined = np.mean(preds, axis=0)
        
        predictions.append((column, test_data['RID'], pred_combined, test_data[column]))
    
    return predictions


data_spm = data
data_mptm = data
pred_24_months_spm = predict_future_values_spm(data_spm, nm_columns.union(biomarker_columns), 5)
for column, rids, preds, actuals in pred_24_months_spm:
    data_spm.loc[data_spm['VISCODE'] == 5, column] = preds
pred_30_months_spm = predict_future_values_spm(data_spm, nm_columns.union(biomarker_columns), 6)


pred_24_months_mptm = predict_future_values_mptm(data_mptm, nm_columns.union(biomarker_columns), 5)
for column, rids, preds, actuals in pred_24_months_mptm:
    data_mptm.loc[data_mptm['VISCODE'] == 5, column] = preds
pred_30_months_mptm = predict_future_values_mptm(data_mptm, nm_columns.union(biomarker_columns), 6)

def count_close_predictions(predictions, tolerance=0.001):
    close_predictions = 0
    total_predictions = 0

    for column, rids, preds, actuals in predictions:
        total_predictions += len(preds)
        close_predictions += np.sum(np.abs(preds - actuals) <= tolerance)

    return close_predictions, total_predictions

# Count close predictions for 24-month and 30-month predictions
close_predictions_24_spm, total_predictions_24_spm = count_close_predictions(pred_24_months_spm)
close_predictions_30_spm, total_predictions_30_spm = count_close_predictions(pred_30_months_spm)

close_predictions_24_mptm, total_predictions_24_mptm = count_close_predictions(pred_24_months_mptm)
close_predictions_30_mptm, total_predictions_30_mptm = count_close_predictions(pred_30_months_mptm)

print(f"SPM - 24-Month Predictions: {close_predictions_24_spm} out of {total_predictions_24_spm} are close to the actual values.")
print(f"SPM - 30-Month Predictions: {close_predictions_30_spm} out of {total_predictions_30_spm} are close to the actual values.")

print(f"MPTM - 24-Month Predictions: {close_predictions_24_mptm} out of {total_predictions_24_mptm} are close to the actual values.")
print(f"MPTM - 30-Month Predictions: {close_predictions_30_mptm} out of {total_predictions_30_mptm} are close to the actual values.")

data_mptm.to_csv('output_predicted_mptm_no_lag.csv', index=False)
data_spm.to_csv('output_predicted_spm_no_lag.csv', index=False)
