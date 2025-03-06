from tqdm import tqdm
import numpy as np
from scipy.optimize import minimize

class LogisticRegression:
    def __init__(self, C=1.0, max_iter=100, class_weight={0: 1, 1: 1}, random_state=None):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.coef_ = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_features = X.shape[1]
        initial_weights = np.zeros(n_features)

        # Weighted loss function
        def loss(weights):
            predictions = self.sigmoid(X.dot(weights))
            class_0_weight = self.class_weight.get(0, 1)
            class_1_weight = self.class_weight.get(1, 1)
            weighted_loss = (
                -class_0_weight * (y == 0) * np.log(1 - predictions) -
                class_1_weight * (y == 1) * np.log(predictions)
            )
            reg = 0.5 / self.C * np.sum(weights**2)
            return np.mean(weighted_loss) + reg

        # Gradient of the loss function
        def grad(weights):
            predictions = self.sigmoid(X.dot(weights))
            error = predictions - y
            weighted_error = (
                self.class_weight.get(0, 1) * (y == 0) * error +
                self.class_weight.get(1, 1) * (y == 1) * error
            )
            reg_grad = weights / self.C
            return X.T.dot(weighted_error) / len(y) + reg_grad

        # Minimize the loss using the Newton-CG method
        result = minimize(
            loss, 
            initial_weights, 
            method='Newton-CG', 
            jac=grad,
            options={'maxiter': self.max_iter}
        )
        self.coef_ = result.x

    def predict_proba(self, X):
        return self.sigmoid(X.dot(self.coef_))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {
            'C': self.C,
            'max_iter': self.max_iter,
            'class_weight': self.class_weight,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
