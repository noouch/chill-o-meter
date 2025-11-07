import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Create a simple test to see what attributes a fitted LDA model has
X = np.random.rand(20, 768)  # 20 samples with 768 features
y = np.array(['A'] * 10 + ['B'] * 10)  # 2 classes

lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X, y)

print("Attributes of a fitted LDA model:")
for attr in dir(lda):
    if not attr.startswith('_'):
        try:
            value = getattr(lda, attr)
            if not callable(value):
                print(f"  {attr}: {type(value)} - {getattr(lda, attr)}")
        except:
            pass

print("\nSpecifically checking for required attributes:")
required_attrs = ['xbar_', 'scalings_', 'means_', 'priors_', 'coef_', 'intercept_', 'classes_']
for attr in required_attrs:
    if hasattr(lda, attr):
        print(f"  {attr}: EXISTS")
    else:
        print(f"  {attr}: MISSING")
