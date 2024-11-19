import pandas as pd

# Load the CSV file
file_path = '/home/justin/personal/bcit/term4/machine-learning-comp4989/insurance/comp4983-project/dim-reduction/old/original_data/trainingset.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Calculate missingness for each column
missingness = data.isnull().mean() * 100

# Display missingness as a percentage for each column
print("Missingness Percentage for Each Column:")
print(missingness)

# Optionally, save the results to a new CSV
missingness.to_csv('missingness_report.csv', header=['Missingness (%)'])
