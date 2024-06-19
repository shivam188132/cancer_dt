import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
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
from sklearn.metrics import accuracy_score

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

# Models to evaluate
models = {
    "LabelPropagation": LabelPropagation(),
    "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "RandomForestClassifier": RandomForestClassifier(),
    "BaggingClassifier": BaggingClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "SVC": SVC(),
    "NuSVC": NuSVC(),
    "NearestCentroid": NearestCentroid(),
    "LogisticRegression": LogisticRegression(),
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis()}

# Evaluate each model
accuracies = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")

# Determine the most accurate model
best_model = max(accuracies, key=accuracies.get)
print(f"\nMost accurate model: {best_model} with accuracy {accuracies[best_model]:.4f}")
