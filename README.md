# Gaussian Naive Bayes Classifier from Scratch

This repository contains a Python implementation of the Gaussian Naive Bayes classifier built from scratch using NumPy. 
The implementation follows the mathematical foundations of the algorithm without relying on scikit-learn's prebuilt implementation (though it uses scikit-learn for dataset loading and evaluation).

## Features

- Pure Python/NumPy implementation of Gaussian Naive Bayes
- Methods for:
  - `fit()` - Training the model on provided data
  - `predict()` - Making predictions on new data
  - Internal `_log_gaussian()` helper for probability calculations
- Example usage with the Breast Cancer Wisconsin dataset
- Accuracy measurement using scikit-learn's metrics

## Implementation Details

The implementation calculates:
- Class means and variances for each feature during training
- Class priors (probabilities)
- Uses log probabilities for numerical stability
- Applies Gaussian probability density function for classification

## Usage Example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and split data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train classifier
clf = GaussianNB().fit(X_train, y_train)

# Make predictions and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

## Requirements

- Python 3.x
- NumPy
- scikit-learn (for dataset and evaluation only)

## Credits

This implementation was inspired by the concepts explained in the YouTube video by NeuralNine:

[Gaussian Naive Bayes Classifier from Scratch](https://www.youtube.com/watch?v=AqW3aPSPIhc&t=1s)

## License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).
