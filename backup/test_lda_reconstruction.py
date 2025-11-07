import json
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the LDA model data
with open("embedding_data/lda_model.json", 'r') as f:
    model_data = json.load(f)

print("Original model data:")
print(f"  Classes: {model_data['classes']}")
print(f"  Coef shape: {np.array(model_data['coef']).shape}")
print(f"  Intercept shape: {np.array(model_data['intercept']).shape}")

# Create a test LDA model by fitting it with dummy data
n_classes = len(model_data["classes"])
n_features = 768
n_samples = 100

# Generate dummy data for fitting
X_dummy = np.random.rand(n_samples, n_features)
y_dummy = np.array([model_data["classes"][i % n_classes] for i in range(n_samples)])

# Fit a model to see what attributes it gets
lda_fitted = LinearDiscriminantAnalysis()
lda_fitted.fit(X_dummy, y_dummy)

print("\nFitted model attributes:")
print(f"  _max_components: {getattr(lda_fitted, '_max_components', 'NOT FOUND')}")
print(f"  explained_variance_ratio_: {getattr(lda_fitted, 'explained_variance_ratio_', 'NOT FOUND')}")

# Try to manually set the parameters
lda_manual = LinearDiscriminantAnalysis()
lda_manual.classes_ = np.array(model_data["classes"])
lda_manual.coef_ = np.array(model_data["coef"])
lda_manual.intercept_ = np.array(model_data["intercept"])
lda_manual.n_features_in_ = n_features
lda_manual.xbar_ = np.zeros(n_features)
lda_manual.scalings_ = np.ones((n_features, 1))
lda_manual.means_ = np.zeros((n_classes, n_features))
lda_manual.priors_ = np.ones(n_classes) / n_classes
lda_manual._fitted = True
lda_manual._max_components = 1

# Test with sample data
test_data = np.random.rand(10, 768)
try:
    result = lda_manual.transform(test_data)
    print(f"\nManual model transform successful! Shape: {result.shape}")
except Exception as e:
    print(f"\nManual model transform failed: {e}")

# Let's also check what the actual model data looks like
print(f"\nActual coef values (first 10): {lda_manual.coef_[0][:10]}")
print(f"Actual intercept values: {lda_manual.intercept_}")
