import json
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the LDA model from file (layer 9)
with open("embedding_data/lda_model_layer9.json", 'r') as f:
    model_data = json.load(f)

print(f"Model data keys: {model_data.keys()}")
print(f"Classes: {model_data['classes']}")
print(f"Coefficient shape: {np.array(model_data['coef']).shape}")
print(f"Intercept shape: {np.array(model_data['intercept']).shape}")

# Create a dummy LDA model to get the right structure
n_classes = len(model_data["classes"])
n_components = min(2, n_classes - 1)

# Create dummy data to fit the model initially
X_dummy = np.random.rand(100, 768)  # 100 samples with 768 features
y_dummy = np.random.choice(model_data["classes"], 100)  # Random labels from the actual classes

# Create and fit a dummy LDA model to get the right structure
lda_dummy = LinearDiscriminantAnalysis(n_components=n_components)
lda_dummy.fit(X_dummy, y_dummy)

print(f"Dummy model coef shape: {lda_dummy.coef_.shape}")
print(f"Dummy model intercept shape: {lda_dummy.intercept_.shape}")

# Now replace the parameters with the saved ones
lda_dummy.classes_ = np.array(model_data["classes"])
lda_dummy.coef_ = np.array(model_data["coef"])
lda_dummy.intercept_ = np.array(model_data["intercept"])

# Test with sample data of correct dimension
test_embeddings = np.random.rand(10, 768)  # 10 samples with 768 features
print(f"Test embeddings shape: {test_embeddings.shape}")

try:
    result = lda_dummy.transform(test_embeddings)
    print(f"LDA transformation successful! Output shape: {result.shape}")
    print(f"Sample output values: {result[:3]}")
except Exception as e:
    print(f"Error in LDA transformation: {e}")
    import traceback
    traceback.print_exc()
