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
print(f"means_: {lda.means_}")
print(f"priors_: {lda.priors_}")
print(f"scalings_: {lda.scalings_}")
print(f"xbar_: {lda.xbar_}")
print(f"explained_variance_ratio_: {lda.explained_variance_ratio_}")

# Test transform
test_data = np.array([[1, 1, 1], [-1, -1, -1]])
transformed = lda.transform(test_data)
decision = lda.decision_function(test_data)
print(f"\nTransform test:")
print(f"Input: {test_data}")
print(f"Transformed: {transformed}")
print(f"Decision function: {decision}")

# Now save all important attributes
model_data = {
    "classes": lda.classes_.tolist(),
    "coef": lda.coef_.tolist(),
    "intercept": lda.intercept_.tolist(),
    "means": lda.means_.tolist(),
    "priors": lda.priors_.tolist(),
    "scalings": lda.scalings_.tolist(),
    "xbar": lda.xbar_.tolist(),
    "explained_variance_ratio": lda.explained_variance_ratio_.tolist(),
    "n_components": lda.n_components,
    "n_features_in": lda.n_features_in_,
    "_max_components": lda._max_components
}

# Save to JSON
import json
with open("complete_lda_model.json", "w") as f:
    json.dump(model_data, f)

# Load back all attributes
with open("complete_lda_model.json", "r") as f:
    loaded_model_data = json.load(f)

# Create new LDA model and load all parameters
lda_loaded = LinearDiscriminantAnalysis(n_components=loaded_model_data["n_components"])
# We need to fit it first to initialize all attributes
lda_loaded.fit(np.random.rand(10, loaded_model_data["n_features_in"]), 
               np.random.choice(loaded_model_data["classes"], 10))

# Now load all the parameters
lda_loaded.classes_ = np.array(loaded_model_data["classes"])
lda_loaded.coef_ = np.array(loaded_model_data["coef"])
lda_loaded.intercept_ = np.array(loaded_model_data["intercept"])
lda_loaded.means_ = np.array(loaded_model_data["means"])
lda_loaded.priors_ = np.array(loaded_model_data["priors"])
lda_loaded.scalings_ = np.array(loaded_model_data["scalings"])
lda_loaded.xbar_ = np.array(loaded_model_data["xbar"])
lda_loaded.explained_variance_ratio_ = np.array(loaded_model_data["explained_variance_ratio"])
lda_loaded.n_features_in_ = loaded_model_data["n_features_in"]
lda_loaded._max_components = loaded_model_data["_max_components"]

print(f"\nLoaded LDA model:")
print(f"coef_: {lda_loaded.coef_}")
print(f"intercept_: {lda_loaded.intercept_}")
print(f"classes_: {lda_loaded.classes_}")
print(f"means_: {lda_loaded.means_}")
print(f"priors_: {lda_loaded.priors_}")
print(f"scalings_: {lda_loaded.scalings_}")
print(f"xbar_: {lda_loaded.xbar_}")
print(f"explained_variance_ratio_: {lda_loaded.explained_variance_ratio_}")

# Test transform with loaded model
transformed_loaded = lda_loaded.transform(test_data)
decision_loaded = lda_loaded.decision_function(test_data)
print(f"\nTransform with loaded model:")
print(f"Input: {test_data}")
print(f"Transformed: {transformed_loaded}")
print(f"Decision function: {decision_loaded}")

# Check if they match
print(f"\nComparison:")
print(f"Original transform: {transformed}")
print(f"Loaded transform: {transformed_loaded}")
print(f"Match exactly: {np.allclose(transformed, transformed_loaded)}")
print(f"Original decision: {decision}")
print(f"Loaded decision: {decision_loaded}")
print(f"Decision match exactly: {np.allclose(decision, decision_loaded)}")
