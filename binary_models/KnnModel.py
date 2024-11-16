# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

# Use KNN for model

# Load the dataset (replace 'your_file.csv' with the actual file path)
data = pd.read_csv("../original_data/trainingset.csv")

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

# Initialize the k-NN classifier (you can tune k for optimal results)
k = 1
knn = KNeighborsClassifier(n_neighbors=k)

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the model using F-score
f_score = f1_score(y_test, y_pred)

# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# r2 = r2_score(y_test, y_pred)
# mape = mean_absolute_percentage_error(y_test, y_pred)
#
# print("Evaluation Metrics for Random Forest Regressor:")
# print(f"Mean Absolute Error (MAE): {mae:.2f}")
# print(f"Mean Squared Error (MSE): {mse:.2f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
# print(f"R-squared (R2): {r2:.2f}")
# print(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")

# F score for best evaluation method
print(f"F-score: {f_score:.4f}")
