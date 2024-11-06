import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from imblearn.over_sampling import SMOTE


# Load the dataset (replace 'your_file.csv' with the actual file path)
data = pd.read_csv("original_data/trainingset.csv")

# Convert output to binary: 0 stays 0, non-zero values become 1
data["ClaimAmount"] = data["ClaimAmount"].apply(lambda x: 0 if x == 0 else 1)

# Separate features and target variable
X = data.drop(columns=["ClaimAmount"])
y = data["ClaimAmount"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Oversample the minority class in the training set
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Define the SVM model with a grid search to find optimal hyperparameters
param_grid = {
    'C': [0.01, 0.001]  # Testing a range of regularization strengths
}

# Initialize SVM with GridSearchCV
svm = LinearSVC(class_weight='balanced', random_state=42)
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_res, y_train_res)

# Find the best model
best_svm = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_svm.predict(X_test)

# Evaluate the model using F-score and additional metrics
f_score = f1_score(y_test, y_pred)
print(f"Best SVM model F-score: {f_score:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Display the best parameters found by GridSearchCV
print("Best parameters found by GridSearchCV:", grid_search.best_params_)