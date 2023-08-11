import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(
        self,
        batch_size=16,
        learning_rate=2e-3,
        num_iter=50,
        regularization=None,
        verbose=1,
    ):
        self.b = batch_size
        self.lr = learning_rate
        self.num_iter = num_iter
        self.regularization = regularization
        self.verbose = verbose

    def loss(self, X: np.ndarray, y: np.ndarray):
        if X.shape[0] == self.l:
            return 1 / X.shape[0] * np.power(np.linalg.norm(X @ self.w - y), 2)
        return (
            1 / X.shape[0] * np.power(np.linalg.norm(X @ self.w[1:] + self.w[0] - y), 2)
        )

    def stochastic_batch_grad(
        self, X: np.ndarray, y: np.ndarray, clipping_thresh: float = 1e10
    ) -> np.ndarray:
        random_indices = np.random.choice(np.arange(self.l), size=self.b)
        X = X[random_indices]
        y = y[random_indices]
        grad = 2 / self.b * X.T.dot(X @ self.w - y)
        while np.linalg.norm(grad) > clipping_thresh:
            grad /= 10
        return grad  # clipping gradient -> gradient explosion prevention

    def fit(self, X: np.ndarray, y: np.ndarray):
        np.set_printoptions(precision=2)
        self.l, self.n = X.shape
        self.w = np.random.rand(self.n + 1, 1)
        upd_X = np.hstack((np.ones((X.shape[0], 1)), X))  # add bias feature
        for i in range(self.num_iter):
            gradient_step = self.stochastic_batch_grad(upd_X, y)
            self.w = self.w - self.lr * gradient_step
            if self.verbose == 2:
                print(f"Weights on iteration {i}: {self.w.squeeze()}")
        self.training_score = self.loss(upd_X, y)
        np.set_printoptions()

    def predict(self, X: np.ndarray):
        return X @ self.w[1:] + self.w[0]

    def get_weights(self):
        return {"w": self.w[1:].squeeze(), "bias": self.w[0].squeeze()}

    def score(self):
        return self.training_score


if __name__ == "__main__":
    # Some tests. More in "check_algorithms.ipynb"

    import matplotlib.pyplot as plt

    l = 100
    X = np.linspace(0, 10, l).reshape((l, 1))
    y = np.array([3 * x - 1 + np.random.normal(loc=0, scale=100) for x in X])

    target = lambda x: 3 * x - 1

    linreg = LinearRegression()
    linreg.fit(X, y)
    pred = linreg.predict(X)

    print(linreg.get_weights())
    plt.scatter(X, y)
    plt.plot(X, target(X.ravel()), label="y", color="r")
    plt.plot(X, pred, label="a", color="green")
    plt.legend()
    plt.show()
