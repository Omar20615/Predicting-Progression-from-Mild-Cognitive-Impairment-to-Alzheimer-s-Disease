from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score
from sklearn.model_selection import cross_val_predict
import pandas as pd
import numpy as np
from tqdm import tqdm

csv_files = ['new_output_data.csv', 'output_predicted_mptm_no_lag_2.csv','output_predicted_mptm_no_lag.csv','output_predicted_spm_no_lag-2.csv','output_predicted_spm_no_lag.csv']  
csv_files2 = ['new_output_data.csv', 'output_predicted_mptm_no_lag.csv','output_predicted_spm_no_lag.csv']  
csv_files3 = ['new_output_data.csv', 'output_predicted_mptm_no_lag_2.csv','output_predicted_spm_no_lag-2.csv']  

results = {}
results2 = {}
results3 = {}


for csv_file in tqdm(csv_files):
    data = pd.read_csv(csv_file)
    X = data.iloc[:, 4:12]  
    y = data['Labels']   
    groups = data['RID']

    svm_clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, probability=True))
    y_pred = cross_val_predict(svm_clf, X, y, cv=GroupKFold(n_splits=5), groups=groups)

    mean_accuracy = np.mean(y == y_pred)
    auc = roc_auc_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = precision_score(y, y_pred)
    mpe = np.mean((y - y_pred) / y)  

    results[csv_file] = {
        'mean_accuracy': mean_accuracy,
        'AUC': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'MPE': mpe
    }
    print(results)


for csv_file, result in results.items():
    print(f"Results for {csv_file}:")
    print(f"Mean accuracy: {result['mean_accuracy']}")
    print(f"AUC: {result['AUC']}")
    print(f"Sensitivity: {result['sensitivity']}")
    print(f"Specificity: {result['specificity']}")
    print(f"Precision: {result['precision']}")
    print(f"MPE: {result['MPE']}")


viscodes = [[1, 2, 3, 4], [1, 2, 3, 4, 5]]

for viscode in viscodes:
    print(f"Predicting for viscode values: {viscode}")
    for csv_file in tqdm(csv_files2):
        data = pd.read_csv(csv_file)
        # Filter data based on viscode values
        data = data[data['VISCODE'].isin(viscode)]
        X = data.iloc[:, 4:12]  
        y = data['Labels']   
        groups = data['RID']

        svm_clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, probability=True))
        y_pred = cross_val_predict(svm_clf, X, y, cv=GroupKFold(n_splits=5), groups=groups)

        mean_accuracy = np.mean(y == y_pred)
        auc = roc_auc_score(y, y_pred)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = precision_score(y, y_pred)
        mpe = np.mean((y - y_pred) / y)  


        results2[f"{csv_file}_viscodes_{','.join(map(str, viscode))}"] = {
        'mean_accuracy': mean_accuracy,
        'AUC': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'MPE': mpe
    }
        print(results2)


    for csv_file,result in results2.items():
        print(f"Results for {csv_file}:")
        print(f"Mean accuracy: {result['mean_accuracy']}")
        print(f"AUC: {result['AUC']}")
        print(f"Sensitivity: {result['sensitivity']}")
        print(f"Specificity: {result['specificity']}")
        print(f"Precision: {result['precision']}")
        print(f"MPE: {result['MPE']}")
              
viscodes = [[1,3],[1,3,5],[1,3,5,6]]

for viscode in viscodes:
    print(f"Predicting for viscode values: {viscode}")
    for csv_file in tqdm(csv_files3):
        data = pd.read_csv(csv_file)
        # Filter data based on viscode values
        data = data[data['VISCODE'].isin(viscode)]
        X = data.iloc[:, 4:12]  
        y = data['Labels']   
        groups = data['RID']

        svm_clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, probability=True))
        y_pred = cross_val_predict(svm_clf, X, y, cv=GroupKFold(n_splits=5), groups=groups)

        mean_accuracy = np.mean(y == y_pred)
        auc = roc_auc_score(y, y_pred)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = precision_score(y, y_pred)
        mpe = np.mean((y - y_pred) / y) 

        results3[f"{csv_file}_viscodes_{','.join(map(str, viscode))}"] = {
        'mean_accuracy': mean_accuracy,
        'AUC': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'MPE': mpe
    }
        print(results3)


    for csv_file, result in results3.items():
        print(f"Results for {csv_file}:")
        print(f"Mean accuracy: {result['mean_accuracy']}")
        print(f"AUC: {result['AUC']}")
        print(f"Sensitivity: {result['sensitivity']}")
        print(f"Specificity: {result['specificity']}")
        print(f"Precision: {result['precision']}")
        print(f"MPE: {result['MPE']}")


df = pd.DataFrame(results).T

print(df)

df.plot(kind='bar', figsize=(12, 6))
plt.ylabel('Score')
plt.title('Metrics for each CSV file')
plt.show()


df2 = pd.DataFrame(results2).T
print(df2)
df2.plot(kind='bar', figsize=(12, 6))
plt.ylabel('Score')
plt.title('Metrics for each CSV file in csv_files2')
plt.show()

df3 = pd.DataFrame(results3).T
print(df3)
df3.plot(kind='bar', figsize=(12, 6))
plt.ylabel('Score')
plt.title('Metrics for each CSV file in csv_files3')
plt.show()

df4 = pd.concat([df, df2, df3])
df4.to_csv('results NM.csv')