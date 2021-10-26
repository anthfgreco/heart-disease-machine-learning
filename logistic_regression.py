import numpy as np

class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        n, d = x.shape

        if self.theta is None:
            self.theta = np.zeros(d)

        for i in range(self.max_iter):
            grad = self._gradient(x, y)
            hess = self._hessian(x)

            self.theta -= self.step_size * np.linalg.inv(hess).dot(grad)

            loss = self._loss(x, y)

            if self.verbose and (i % 100) == 0:
                print("Loss: ", loss, "Theta: ", self.theta)

        if self.verbose:
            print("Final theta is: ", self.theta)

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        return self._sigmoid(x.dot(self.theta))
  
    def _loss(self, x, y):
        eps = 1e-10
        h = self._sigmoid(x.dot(self.theta))
        loss = - np.mean(y * np.log(h + eps) + (1 - y)*np.log(1 - h + eps))
        return loss

    def _gradient(self, x, y):
        n, d = x.shape
        h = self._sigmoid(x.dot(self.theta))
        grad = 1/n * x.T.dot(h-y)
        return grad

    def _hessian(self, x):
        n, d = x.shape
        h = self._sigmoid(x.dot(self.theta))
        diag = np.diag(h*(1-h))
        hess = 1/n * x.T.dot(diag).dot(x)
        return hess

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))