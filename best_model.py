import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Load the data
file_path = 'Cancer_dataset_1500_.xlsx'
data = pd.read_excel(file_path)

# Features and target
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# List of models to evaluate
models = [
    ("GradientBoostingClassifier", GradientBoostingClassifier()),
    ("RandomForestClassifier", RandomForestClassifier()),
    ("BaggingClassifier", BaggingClassifier()),
    ("AdaBoostClassifier", AdaBoostClassifier()),
    ("ExtraTreesClassifier", ExtraTreesClassifier()),
    ("LogisticRegression", LogisticRegression()),
    ("DecisionTreeClassifier", DecisionTreeClassifier()),
    ("SVC", SVC(probability=True)),
    ("GaussianNB", GaussianNB()),
    ("KNeighborsClassifier", KNeighborsClassifier())
]

# Evaluate each model
for model_name, model in models:
    model.fit(X_train_scaled, y_train)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    thresholds = np.arange(0.3, 0.61, 0.01)
    metrics = {'Threshold': [], 'False Negatives': []}
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics['Threshold'].append(threshold)
        metrics['False Negatives'].append(fn)
    
    metrics_df = pd.DataFrame(metrics)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(metrics_df, x='Threshold', weights='False Negatives', bins=len(thresholds), kde=True)
    plt.xlabel('Threshold')
    plt.ylabel('Count of False Negatives')
    plt.title(f'Histogram of False Negatives for Different Thresholds (0.3 to 0.6) - {model_name}')
    plt.axvline(0.34, color='r', linestyle='--', label='Optimal Threshold: 0.34')
    plt.legend()
    plt.show()
