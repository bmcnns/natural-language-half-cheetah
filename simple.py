import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Load data
df = pd.read_csv('half_cheetah_descriptions.csv')

# Convert descriptions to TF-IDF features
vectorizer = TfidfVectorizer(max_features=300)  # Adjust max_features for your dataset size
X_text = vectorizer.fit_transform(df['description']).toarray()
y_coords = df[['x0', 'x1']].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y_coords, test_size=0.2, random_state=42
)

# Train a LightGBM model for each coordinate
models = {}
for i, coord in enumerate(['x0', 'x1']):
    models[coord] = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    models[coord].fit(X_train, y_train[:, i])

# Predict on the test set
y_pred = np.column_stack([models['x0'].predict(X_test), models['x1'].predict(X_test)])

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")

# Predict coordinates for a new prompt
new_prompt = ["This elite uses both legs equally."]
X_new = vectorizer.transform(new_prompt).toarray()
predicted_coords = [
    models['x0'].predict(X_new)[0],
    models['x1'].predict(X_new)[0]
]

print(f"Predicted coordinates: x0 = {predicted_coords[0]}, x1 = {predicted_coords[1]}")


with open('label_embedder.pkl', 'wb') as f:
    pickle.dump((models, vectorizer), f)
