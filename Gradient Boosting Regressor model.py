import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'QuantHockey Dataset - Less Features LR.csv'  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Prepare the data by excluding non-predictive features
predictive_features = data.drop(columns=['Rk', 'Year', 'FantasyPoints', 'NextYearFPs'])

# Target variable
target = data['NextYearFPs']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(predictive_features, target, test_size=0.2, random_state=42)

# Initialize a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler on the training data and transform both the training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Gradient Boosting Regressor
gradient_boosting_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
gradient_boosting_model.fit(X_train_scaled, y_train)

# Predict on the test data
y_pred_gb = gradient_boosting_model.predict(X_test_scaled)

# Calculate the performance metrics
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

# Output the performance metrics
print(f'Mean Squared Error (MSE): {mse_gb}')
print(f'R-squared (R²): {r2_gb}')
