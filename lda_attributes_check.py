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

print("LDA model attributes:")
for attr in dir(lda):
    if not attr.startswith('_') or attr in ['_fitted']:
        try:
            value = getattr(lda, attr)
            if not callable(value):
                print(f"{attr}: {value} (type: {type(value)})")
        except:
            pass

print(f"\nDetailed coefficient info:")
print(f"coef_ shape: {lda.coef_.shape}")
print(f"coef_ dtype: {lda.coef_.dtype}")
print(f"intercept_ shape: {lda.intercept_.shape}")
print(f"intercept_ dtype: {lda.intercept_.dtype}")

# Check if there are any private attributes that might be important
print(f"\nPrivate attributes:")
for attr in dir(lda):
    if attr.startswith('_') and not attr.startswith('__'):
        try:
            value = getattr(lda, attr)
            if not callable(value):
                print(f"{attr}: {value} (type: {type(value)})")
        except:
            print(f"{attr}: <error accessing>")

# Test transform
test_data = np.array([[1, 1, 1], [-1, -1, -1]])
transformed = lda.transform(test_data)
print(f"\nTransform test:")
print(f"Input: {test_data}")
print(f"Transformed: {transformed}")

# Try to understand what's happening by looking at the decision function
decision = lda.decision_function(test_data)
print(f"\nDecision function:")
print(f"Decision: {decision}")

# Check if decision function matches our manual computation
print(f"\nManual computation vs decision function:")
for i, sample in enumerate(test_data):
    manual = np.dot(sample, lda.coef_[0]) + lda.intercept_[0]
    print(f"Sample {i}: manual={manual}, decision={decision[i]}, diff={abs(manual - decision[i])}")
