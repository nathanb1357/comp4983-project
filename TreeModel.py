import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE

# Load the dataset (replace 'your_file.csv' with the actual file path)
data = pd.read_csv("original_data/trainingset.csv")

# Convert output to binary: 0 stays 0, non-zero values become 1
data["ClaimAmount"] = data["ClaimAmount"].apply(lambda x: 0 if x == 0 else 1)

# Separate features and target variable
X = data.drop(columns=["ClaimAmount"])
y = data["ClaimAmount"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Oversample the minority class in the training set
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Initialize RandomForest with GridSearchCV
rf = RandomForestClassifier(class_weight='balanced_subsample', random_state=42)
rf.fit(X_train_res, y_train_res)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Evaluate the model using F-score and additional metrics
f_score = f1_score(y_test, y_pred)
print(f"Best Random Forest model F-score: {f_score:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
