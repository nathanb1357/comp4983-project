# Basic model to improve repeatedly over the term

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
all_data = pd.read_csv("./original_data/trainingset.csv")
claim_data = pd.read_csv("./edited_data/non_zero_claims.csv")


# Convert ClaimAmount to binary for classification in Stage 1
all_data["HasClaim"] = all_data["ClaimAmount"].apply(lambda x: 0 if x == 0 else 1)

# Separate features and target variables
X_all = all_data.drop(columns=["ClaimAmount", "HasClaim"])
y_all_classification = all_data["HasClaim"]
y_all_regression = all_data["ClaimAmount"]

# Split dataset for both classification and regression stages
# Split dataset for both classification and regression stages
X_all_train, X_all_test, y_class_train, y_class_test, y_regression_train, y_regression_test = train_test_split(
    X_all, y_all_classification, y_all_regression, test_size=0.25, random_state=42
)

# Scale features
scaler = StandardScaler()
X_all_train = scaler.fit_transform(X_all_train)
X_all_test = scaler.transform(X_all_test)


# Stage 1 - Classification Model to predict if a claim exists
classifier = RandomForestClassifier(class_weight="balanced_subsample", random_state=42)
# classifier = LogisticRegression(C=0.1, penalty="l2", class_weight="balanced", solver="saga", random_state=42)
# classifier = KNeighborsClassifier(metric='euclidean', n_neighbors=1)
classifier.fit(X_all_train, y_class_train)

# Predict the existence of a claim
y_class_pred = classifier.predict(X_all_test)

# Evaluate Classification Model
f1 = f1_score(y_class_test, y_class_pred)
print(f"Classification F1 Score: {f1:.4f}")


# Stage 2 - Regression Model to predict claim amount if a claim exists
# Filter the training and testing data where a claim is predicted
X_claim = claim_data.drop(columns=["ClaimAmount"])
y_claim = claim_data["ClaimAmount"]

X_claim_train, _, y_claim_train, _ = train_test_split(X_claim, y_claim, test_size=0.2)

# Apply scaling for regression (if needed)
scaler = StandardScaler()
X_claim_train = scaler.fit_transform(X_claim_train)

# Initialize the Regression Model
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_claim_train, y_claim_train)

X_test_claims = X_all_test[y_class_pred == 1]
y_test_claims = y_regression_test[y_class_pred == 1]

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