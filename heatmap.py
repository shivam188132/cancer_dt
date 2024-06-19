import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'Cancer_dataset_1500_.xlsx'
data = pd.read_excel(file_path)

# List of features used in the model
features = ['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']

# Filter the data to include only these features and the target variable
data_filtered = data[features + ['Diagnosis']]

# Compute the correlation matrix
corr_matrix = data_filtered.corr()

# Plot the heatmap for correlations with 'Diagnosis'
plt.figure(figsize=(8, 10))
sns.heatmap(corr_matrix[['Diagnosis']].sort_values(by='Diagnosis', ascending=False), annot=True, cmap='Blues', vmin=-1, vmax=1)
plt.title('Correlation with Diagnosis', color='blue', fontsize=12)
plt.show()
