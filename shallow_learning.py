import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

df = pd.read_csv('half_cheetah_descriptions.csv')

vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(df['description'])
y_coords = df[['x0', 'x1']].values

model = LinearRegression()
model.fit(X_text, y_coords)

new_prompt = [
    "This elite hops dominantly on its front leg"]

# Transform the prompt using the vectorizer
X_test = vectorizer.transform(new_prompt)

# Predict the coordinates using the trained model
predicted_coords = model.predict(X_test)

# Print the predicted coordinates
print(f"Predicted x, y coordinates: {predicted_coords[0]}")
