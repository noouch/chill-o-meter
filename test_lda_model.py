import json
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Load the LDA model
with open("embedding_data/lda_model.json", 'r') as f:
    model_data = json.load(f)

# Create LDA model with the same parameters
n_classes = len(model_data["classes"])
n_components = min(2, n_classes - 1)
lda = LinearDiscriminantAnalysis(n_components=n_components)

# Set the model parameters
lda.classes_ = np.array(model_data["classes"])
lda.coef_ = np.array(model_data["coef"])
lda.intercept_ = np.array(model_data["intercept"])

# For scikit-learn compatibility, set additional required attributes
lda._explained_variance_ratio = np.ones(n_components) / n_components
lda.means_ = np.zeros((n_classes, lda.coef_.shape[1]))
lda.priors_ = np.ones(n_classes) / n_classes
lda.scalings_ = np.ones((n_classes, lda.coef_.shape[1]))
lda.xbar_ = np.zeros(lda.coef_.shape[1])
lda.n_features_in_ = lda.coef_.shape[1]

print(f"LDA model loaded with classes: {model_data['classes']}")
print(f"Coefficient shape: {lda.coef_.shape}")
print(f"Expected input features: {lda.coef_.shape[1]}")

# Test with sample data of correct dimension
test_embeddings = np.random.rand(10, 768)  # 10 samples with 768 features
print(f"Test embeddings shape: {test_embeddings.shape}")

try:
    result = lda.transform(test_embeddings)
    print(f"LDA transformation successful! Output shape: {result.shape}")
except Exception as e:
    print(f"Error in LDA transformation: {e}")
