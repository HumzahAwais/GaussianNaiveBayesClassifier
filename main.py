import numpy as np
# Just the dataset and stuff needed to test the code
# Not any prebuilt stuff
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class GaussianNB:

    # Training the Gaussian Naive Bayes model
    def fit(self, X, Y):
        # Turn the data into numpy arrays
        X, Y = np.asarray(X), np.asarray(Y)

        # The classes that you can have
        self.classes_ = np.unique(Y)

        # The number of classes and features
        # X.shape[1] - index 0 is the number of instances and index 1 is the number of features per instance
        n_classes, n_features = len(self.classes_), X.shape[1]

        # Empty numpy arrays
        # For the means and variances and priors
        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

        # Iterate through each class
        for idx, k in enumerate(self.classes_):
            # Create matrix Xk which has all input data
            # where correct classification is k
            Xk = X[Y == k]

            # Calculate the stuff
            self.means_[idx] = Xk.mean(axis=0)
            self.variances_[idx] = Xk.var(axis=0)
            self.priors_[idx] = Xk.shape[0] / X.shape[0]
        
        return self
    
    # Log function for Gaussian Naive Bayes
    def _log_gaussian(self, X):
        # Calculate the first part
        num = -0.5 * (X[:, None, :] - self.means_)**2 / self.variances_
        # Calculate the log probability
        log_prob = num - 0.5 * np.log(2 * np.pi * self.variances_)
        # Return the sum
        return log_prob.sum(axis=2)
    
    def predict(self, X):
        # Turn the input data into a numpy array
        X = np.asarray(X)

        # Calculate the log probablity 
        log_likelihood = self._log_gaussian(X)

        # Calculate the log prior
        log_prior = np.log(self.priors_)

        # We want to maximise log_likelihood + log_prior
        # and return the class with the highest value
        # np.argmax returns the index of the maximum value along the specified axis
        return self.classes_[np.argmax(log_likelihood + log_prior, axis=1)]

# Load dataset and split it into training and testing sets
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create classifier instance and fit it to the training data
clf = GaussianNB().fit(X_train, y_train)

# Predict class for testing data
y_pred = clf.predict(X_test)

# Print accuracy of the predictions
print("Accuracy:", accuracy_score(y_test, y_pred))