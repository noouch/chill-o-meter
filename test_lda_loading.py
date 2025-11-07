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

# Create LDA model with the same parameters
n_classes = len(model_data["classes"])
n_components = min(2, n_classes - 1)
lda = LinearDiscriminantAnalysis(n_components=n_components)

# Set the model parameters
lda.classes_ = np.array(model_data["classes"])
lda.coef_ = np.array(model_data["coef"])
lda.intercept_ = np.array(model_data["intercept"])

# For scikit-learn compatibility, set additional required attributes
lda.n_features_in_ = lda.coef_.shape[1]

# Set other required attributes for a fitted LDA model
lda.xbar_ = np.zeros(lda.n_features_in_)
lda.scalings_ = np.ones((lda.n_features_in_, n_components))
lda.means_ = np.zeros((n_classes, lda.n_features_in_))
lda.priors_ = np.ones(n_classes) / n_classes
lda._max_components = n_components
lda.explained_variance_ratio_ = np.ones(n_components) / n_components

# Mark as fitted
lda._fitted = True

print(f"LDA model loaded with classes: {model_data['classes']}")
print(f"Expected input features: {lda.n_features_in_}")

# Test with sample data of correct dimension
test_embeddings = np.random.rand(10, 768)  # 10 samples with 768 features
print(f"Test embeddings shape: {test_embeddings.shape}")

try:
    result = lda.transform(test_embeddings)
    print(f"LDA transformation successful! Output shape: {result.shape}")
    print(f"Sample output values: {result[:3]}")
except Exception as e:
    print(f"Error in LDA transformation: {e}")
