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

print(f"\nLDA model attributes after manual setting:")
print(f"_fitted: {hasattr(lda, '_fitted') and lda._fitted}")
print(f"classes_: {hasattr(lda, 'classes_') and lda.classes_}")
print(f"coef_: {hasattr(lda, 'coef_') and lda.coef_.shape}")
print(f"intercept_: {hasattr(lda, 'intercept_') and lda.intercept_.shape}")
print(f"xbar_: {hasattr(lda, 'xbar_') and lda.xbar_.shape}")
print(f"scalings_: {hasattr(lda, 'scalings_') and lda.scalings_.shape}")
print(f"means_: {hasattr(lda, 'means_') and lda.means_.shape}")
print(f"priors_: {hasattr(lda, 'priors_') and lda.priors_.shape}")
print(f"n_features_in_: {hasattr(lda, 'n_features_in_') and lda.n_features_in_}")

# Now let's create a properly fitted LDA model for comparison
# We'll need some dummy data to fit it
X = np.random.rand(100, 768)  # 100 samples with 768 features
y = np.random.choice(['A', 'B'], 100)  # Random labels

lda_fitted = LinearDiscriminantAnalysis(n_components=n_components)
lda_fitted.fit(X, y)

print(f"\nFitted LDA model attributes:")
print(f"_fitted: {hasattr(lda_fitted, '_fitted') and lda_fitted._fitted}")
print(f"classes_: {hasattr(lda_fitted, 'classes_') and lda_fitted.classes_}")
print(f"coef_: {hasattr(lda_fitted, 'coef_') and lda_fitted.coef_.shape}")
print(f"intercept_: {hasattr(lda_fitted, 'intercept_') and lda_fitted.intercept_.shape}")
print(f"xbar_: {hasattr(lda_fitted, 'xbar_') and lda_fitted.xbar_.shape}")
print(f"scalings_: {hasattr(lda_fitted, 'scalings_') and lda_fitted.scalings_.shape}")
print(f"means_: {hasattr(lda_fitted, 'means_') and lda_fitted.means_.shape}")
print(f"priors_: {hasattr(lda_fitted, 'priors_') and lda_fitted.priors_.shape}")
print(f"n_features_in_: {hasattr(lda_fitted, 'n_features_in_') and lda_fitted.n_features_in_}")

# Compare coefficient values
print(f"\nCoefficient comparison:")
print(f"Manual coef_: {lda.coef_[:1, :5]}")  # First 5 values of first row
print(f"Fitted coef_: {lda_fitted.coef_[:1, :5]}")  # First 5 values of first row

# Compare intercept values
print(f"\nIntercept comparison:")
print(f"Manual intercept_: {lda.intercept_}")
print(f"Fitted intercept_: {lda_fitted.intercept_}")
