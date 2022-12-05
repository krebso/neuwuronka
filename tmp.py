import numpy as np

class Linear:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.grad_w = np.zeros_like(w)
        self.grad_b = np.zeros_like(b)

    def forward(self, x):
        return np.dot(self.w, x) + self.b

    def backward(self, z, grad):
        grad_b += grad
        grad_w += grad.transpose() * z
        return self.w.transpose() * grad

class ReLU:
    def __init__(self):
        self.activation = None
        self.activation_prime = None

    def forward(self, x):
        self.activation = np.maximum(x, 0)
        self.activation_prime = np.where(x > 0, 1, 0)
        return self.activation

    def backward(self, act, grad):
        return grad * self.activation_prime


class Softmax:
    def __init__(self):
        self.activation = None

    def forward(self, x):
        exp = np.exp(x)
        self.activation = exp / np.sum(exp)

    def backward(self, act, grad):
        return act - grad

class Model:
    def __init__(self) -> None:
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, x, y):
        for layer in reversed(self.layers):
            x = layer.backward(x, y)
        return x


w1 = np.array([[-0.516964, 1.2219212], [0.869636, 1.6182174]])
b1 = np.array([[0.721333, 1.588556]])

w2 = np.array([[-0.496103, 1.238407], [0.666369, 1.473580]])
b2 = np.array([[0.742194, 1.385290]])

model = Model()
model.add(Linear(w1, b1))
model.add(ReLU())
model.add(Linear(w2, b2))
model.add(ReLU)
model.add(Softmax())

model.forward(np.array([1, 1]))