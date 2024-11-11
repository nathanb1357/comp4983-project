# Basic model to improve repeatedly over the term

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("./original_data/trainingset.csv")

# Convert ClaimAmount to binary for classification in Stage 1
data["HasClaim"] = data["ClaimAmount"].apply(lambda x: 0 if x == 0 else 1)

# Separate features and target variables
X = data.drop(columns=["ClaimAmount", "HasClaim"])
y_classification = data["HasClaim"]
y_regression = data["ClaimAmount"]

# Split dataset for both classification and regression stages
X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_classification, test_size=0.25, random_state=42)

# Stage 1 - Classification Model to predict if a claim exists
classifier = RandomForestClassifier(class_weight="balanced_subsample", random_state=42)
classifier.fit(X_train, y_class_train)

# Predict the existence of a claim
y_class_pred = classifier.predict(X_test)

# Evaluate Classification Model
f1 = f1_score(y_class_test, y_class_pred)
print(f"Classification F1 Score: {f1:.4f}")

# Stage 2 - Regression Model to predict claim amount if a claim exists
# Filter the training and testing data where a claim is predicted
claim_data = pd.read_csv("./edited_data/non_zero_claims.csv")
X_claim = claim_data.drop(columns=["ClaimAmount"])
y_claim = claim_data["ClaimAmount"]

X_train_claims, X_test_claims, y_train_claims, y_test_claims = train_test_split(X_claim, y_claim, test_size=0.2)

# Apply scaling for regression (if needed)
scaler = StandardScaler()
X_train_claims = scaler.fit_transform(X_train_claims)
X_test_claims = scaler.transform(X_test_claims)

# Initialize the Regression Model
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train_claims, y_train_claims)

# Predict claim amounts on instances with predicted claims
y_pred_claim_amount = regressor.predict(X_test_claims)

# Evaluate Regression Model
mae = mean_absolute_error(y_test_claims, y_pred_claim_amount)
mse = mean_squared_error(y_test_claims, y_pred_claim_amount)
rmse = mse ** 0.5
r2 = r2_score(y_test_claims, y_pred_claim_amount)

print("\nRegression Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")