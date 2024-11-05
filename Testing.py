from Dataset import Dataset
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


ds = Dataset("original_data/trainingset.csv")

ds.create_train_test(ratio=0.75)
ds.define_label_features("ClaimAmount")

model = RandomForestRegressor()
model.fit(ds.train_features, ds.train_label)
test_predictions = model.predict(ds.test_features)


mae = mean_absolute_error(ds.test_label, test_predictions)
mse = mean_squared_error(ds.test_label, test_predictions)
rmse = mean_squared_error(ds.test_label, test_predictions, squared=False)
r2 = r2_score(ds.test_label, test_predictions)
mape = mean_absolute_percentage_error(ds.test_label, test_predictions)

print("Evaluation Metrics for Random Forest Regressor:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")

