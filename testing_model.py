import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV, RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data
file_path = '/mnt/data/Cancer_dataset_1500_.xlsx'
df = pd.read_excel(file_path)

# Features and target
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models to evaluate
models = {
    "LabelPropagation": LabelPropagation(),
    "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "ExtraTreeClassifier": ExtraTreeClassifier(),
    "ExtraTreesClassifier": ExtraTreesClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "BaggingClassifier": BaggingClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "SVC": SVC(probability=True),
    "NuSVC": NuSVC(probability=True),
    "NearestCentroid": NearestCentroid(),
    "LogisticRegression": LogisticRegression(),
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
    "CalibratedClassifierCV": CalibratedClassifierCV(),
    "RidgeClassifierCV": RidgeClassifierCV(),
    "RidgeClassifier": RidgeClassifier(),
    "BernoulliNB": BernoulliNB(),
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
    "SGDClassifier": SGDClassifier(),
    "GaussianNB": GaussianNB(),
    "PassiveAggressiveClassifier": PassiveAggressiveClassifier(),
    "Perceptron": Perceptron(),
    "DummyClassifier": DummyClassifier()
}

# Evaluate each model
accuracies = {}
for name, model in models.items():
    try:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"{name} could not be evaluated due to an error: {e}")

# Determine the most accurate model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
print(f"\nMost accurate model: {best_model_name} with accuracy {accuracies[best_model_name]:.4f}")

# Save the best model to a pkl file
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print("Best model saved to 'best_model.pkl'")

# Now let's plot the histogram of false negatives for the best model
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
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
plt.title(f'Histogram of False Negatives for Different Thresholds (0.3 to 0.6) - {best_model_name}')
plt.axvline(0.34, color='r', linestyle='--', label='Optimal Threshold: 0.34')
plt.legend()
plt.show()
