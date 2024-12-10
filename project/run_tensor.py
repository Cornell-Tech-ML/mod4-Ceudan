"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

# TODO: Implement for Task 2.5.
class Network(minitorch.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.layer1 = LinearLayer(2, hidden)
        self.layer2 = LinearLayer(hidden, hidden)
        self.layer3 = LinearLayer(hidden, 1)

    def forward(self, inputs):
        # z1 = self.layer1.forward(inputs).relu()
        # z2 = self.layer2.forward(z1).relu()
        z1 = self.layer1.forward(inputs).sigmoid()
        z2 = self.layer2.forward(z1).sigmoid()
        z3 = self.layer3.forward(z2).sigmoid()
        return z3

class LinearLayer(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)

    def forward(self, inputs):
        inputs = inputs.view(*inputs.shape, 1)
        in_mul_weights = inputs * self.weights.value
        in_mul_weights_sum = in_mul_weights.sum(1)
        in_mul_weights_add_bias = in_mul_weights_sum + self.bias.value
        result = in_mul_weights_add_bias.view(in_mul_weights_add_bias.shape[0], in_mul_weights_add_bias.shape[2])
        return result


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 5
    RATE = 0.5
    MAX_EPOCHS = 1
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data=data, learning_rate=RATE,max_epochs=MAX_EPOCHS)
