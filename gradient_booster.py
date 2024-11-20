import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv('half_cheetah_descriptions.csv')

# Preprocess text data using TF-IDF
vectorizer = TfidfVectorizer(
    max_features=500, ngram_range=(1, 2), stop_words='english')
X_text = vectorizer.fit_transform(df['description'])

# Extract target coordinates
y_coords = df[['x0', 'x1']].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y_coords, test_size=0.2, random_state=42)

# Train a Gradient Boosting Regressor for each target coordinate (x0 and x1)
model_x0 = GradientBoostingRegressor(random_state=42)
model_x1 = GradientBoostingRegressor(random_state=42)

model_x0.fit(X_train, y_train[:, 0])
model_x1.fit(X_train, y_train[:, 1])

# Make predictions on the test set
y_pred_x0 = model_x0.predict(X_test)
y_pred_x1 = model_x1.predict(X_test)

# Combine predictions
y_pred = list(zip(y_pred_x0, y_pred_x1))

# Evaluate the model
mse_x0 = mean_squared_error(y_test[:, 0], y_pred_x0)
mse_x1 = mean_squared_error(y_test[:, 1], y_pred_x1)
r2_x0 = r2_score(y_test[:, 0], y_pred_x0)
r2_x1 = r2_score(y_test[:, 1], y_pred_x1)

print(f"Performance for x0: MSE = {mse_x0}, R² = {r2_x0}")
print(f"Performance for x1: MSE = {mse_x1}, R² = {r2_x1}")

# New prompt for prediction
new_prompt = [
    "This elite is hopping entirely on its front leg"
]
X_new = vectorizer.transform(new_prompt)

# Predict coordinates for the new prompt
predicted_x0 = model_x0.predict(X_new)
predicted_x1 = model_x1.predict(X_new)

print(f"Predicted coordinates: x0 = {predicted_x0[0]}, x1 = {predicted_x1[0]}")
