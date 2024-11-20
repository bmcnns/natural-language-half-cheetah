import pickle

def predict_coordinates_from_label(models, vectorizer, label):
    transformed_label = vectorizer.transform([label]).toarray()
    return [
        models['x0'].predict(transformed_label)[0],
        models['x1'].predict(transformed_label)[0]
    ]

if __name__ == '__main__':
    model_path = 'models/half_cheetah_label_embedder.pkl'

    with open(model_path, 'rb') as f:
        models, vectorizer = pickle.load(f)
        
    while True:
        print("Enter a prompt or type 'stop' to stop.")
        prompt = input()

        if prompt == 'stop':
            break

        coordinates = predict_coordinates_from_label(models, vectorizer, prompt)
        print(f"This elite would map to the point {coordinates}\n")
