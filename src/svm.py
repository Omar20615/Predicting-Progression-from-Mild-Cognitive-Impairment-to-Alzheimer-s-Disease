import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

data = pd.read_csv("output_complete.csv")

X = data.iloc[:, 4:]  
y = data['Labels']    
p_values = []

for biomarker in tqdm(X.columns):
    MCIp_values = X[y == 1][biomarker]
    MCIs_values = X[y == 0][biomarker]
    _, p_value = ttest_ind(MCIp_values, MCIs_values)
    p_values.append(p_value)

sorted_indices = np.argsort(p_values)
model_ranks = range(1, len(X.columns) + 1)
best_accuracy = 0
best_subset = None

group_kfold = GroupKFold(n_splits=5)
groups = data['RID']

for rank in tqdm(model_ranks):
    selected_features = X.columns[sorted_indices[:rank]]
    svm_clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, probability=True))
    accuracies = cross_val_score(svm_clf, X[selected_features], y, cv=group_kfold, groups=groups)
    mean_accuracy = np.mean(accuracies)

    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_subset = selected_features

print("Best subset of biomarkers:", best_subset)
print("Best accuracy:", best_accuracy)