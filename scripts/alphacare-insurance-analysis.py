import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data_path = 'data/insurance.csv'
data = pd.read_csv(data_path)

# Create a directory to store plots
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# Univariate Analysis
# Numerical Columns: Plot histograms
numerical_columns = ['Total_Claim', 'Premium']
for column in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data[column], kde=True, bins=20, color='blue')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'plots/histogram_{column}.png')
    plt.show()

# Categorical Columns: Plot bar charts
categorical_columns = ['Province', 'Zipcode', 'Gender']
for column in categorical_columns:
    plt.figure(figsize=(8, 5))
    data[column].value_counts().plot(kind='bar', color='orange', edgecolor='black')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'plots/bar_chart_{column}.png')
    plt.show()
