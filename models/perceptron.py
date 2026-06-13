import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, epoch=50):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.weights = None
        self.bias = 0
        self.errors = []

    def activation(self, x):
        return 1 if x >= 0 else 0

    def fit(self, X, y):
        jumlah_fitur = X.shape[1]

        # Inisialisasi bobot awal = 0
        self.weights = np.zeros(jumlah_fitur)
        self.bias = 0

        for _ in range(self.epoch):
            total_error = 0

            for xi, target in zip(X, y):
                linear_output = np.dot(xi, self.weights) + self.bias
                prediction = self.activation(linear_output)

                error = target - prediction

                # Update bobot dan bias
                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error

                total_error += abs(error)

            self.errors.append(total_error)

    def predict_one(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        return self.activation(linear_output)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])
