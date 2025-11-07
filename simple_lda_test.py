import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Create simple test data
np.random.seed(42)
X_class1 = np.random.randn(50, 3) + [2, 2, 2]  # Class 1 centered at [2, 2, 2]
X_class2 = np.random.randn(50, 3) + [-2, -2, -2]  # Class 2 centered at [-2, -2, -2]
X = np.vstack([X_class1, X_class2])
y = np.hstack([np.zeros(50), np.ones(50)])

# Fit LDA model
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X, y)

print("Original LDA model:")
print(f"coef_: {lda.coef_}")
print(f"intercept_: {lda.intercept_}")
print(f"classes_: {lda.classes_}")

# Test transform
test_data = np.array([[1, 1, 1], [-1, -1, -1]])
transformed = lda.transform(test_data)
print(f"\nTransform test:")
print(f"Input: {test_data}")
print(f"Transformed: {transformed}")

# Manual computation
print(f"\nManual computation:")
for i, sample in enumerate(test_data):
    manual = np.dot(sample, lda.coef_[0]) + lda.intercept_[0]
    print(f"Sample {i}: manual={manual}, sklearn={transformed[i]}, diff={abs(manual - transformed[i])}")

# Now save and reload the model like we do in our code
model_data = {
    "classes": lda.classes_.tolist(),
    "coef": lda.coef_.tolist(),
    "intercept": lda.intercept_.tolist()
}

# Save to JSON (simulating our process)
import json
with open("test_lda_model.json", "w") as f:
    json.dump(model_data, f)

# Load back like we do in our code
with open("test_lda_model.json", "r") as f:
    loaded_model_data = json.load(f)

# Create new LDA model and load parameters
lda_loaded = LinearDiscriminantAnalysis(n_components=1)
lda_loaded.fit(np.random.rand(10, 3), np.random.choice([0, 1], 10))  # Dummy fit
lda_loaded.classes_ = np.array(loaded_model_data["classes"])
lda_loaded.coef_ = np.array(loaded_model_data["coef"])
lda_loaded.intercept_ = np.array(loaded_model_data["intercept"])

print(f"\nLoaded LDA model:")
print(f"coef_: {lda_loaded.coef_}")
print(f"intercept_: {lda_loaded.intercept_}")
print(f"classes_: {lda_loaded.classes_}")

# Test transform with loaded model
transformed_loaded = lda_loaded.transform(test_data)
print(f"\nTransform with loaded model:")
print(f"Input: {test_data}")
print(f"Transformed: {transformed_loaded}")

# Check if they match
print(f"\nComparison:")
print(f"Original transform: {transformed}")
print(f"Loaded transform: {transformed_loaded}")
print(f"Match exactly: {np.allclose(transformed, transformed_loaded)}")
