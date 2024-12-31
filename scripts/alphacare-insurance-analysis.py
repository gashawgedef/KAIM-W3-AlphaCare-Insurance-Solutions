# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the data
# data_path = '../data/insurance.csv'
# data = pd.read_csv(data_path)

# # Create a directory to store plots
# import os
# if not os.path.exists('plots'):
#     os.makedirs('plots')

# # Univariate Analysis
# # Numerical Columns: Plot histograms
# numerical_columns = ['Total_Claim', 'Premium']
# for column in numerical_columns:
#     plt.figure(figsize=(8, 5))
#     sns.histplot(data[column], kde=True, bins=20, color='blue')
#     plt.title(f'Distribution of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Frequency')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.savefig(f'plots/histogram_{column}.png')
#     plt.show()

# # Categorical Columns: Plot bar charts
# categorical_columns = ['Province', 'Zipcode', 'Gender']
# for column in categorical_columns:
#     plt.figure(figsize=(8, 5))
#     data[column].value_counts().plot(kind='bar', color='orange', edgecolor='black')
#     plt.title(f'Distribution of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Count')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.savefig(f'plots/bar_chart_{column}.png')
#     plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path):
    """
    Load the dataset from the specified file path.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def create_directory(directory):
    """
    Create a directory if it does not already exist.

    Args:
        directory (str): Path of the directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_numerical_distributions(data, columns, output_dir):
    """
    Plot histograms for numerical columns.

    Args:
        data (pd.DataFrame): The dataset.
        columns (list): List of numerical column names.
        output_dir (str): Directory to save the plots.
    """
    for column in columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data[column], kde=True, bins=20, color='blue')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/histogram_{column}.png')
        plt.show()

def plot_categorical_distributions(data, columns, output_dir):
    """
    Plot bar charts for categorical columns.

    Args:
        data (pd.DataFrame): The dataset.
        columns (list): List of categorical column names.
        output_dir (str): Directory to save the plots.
    """
    for column in columns:
        plt.figure(figsize=(8, 5))
        data[column].value_counts().plot(kind='bar', color='orange', edgecolor='black')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/bar_chart_{column}.png')
        plt.show()
