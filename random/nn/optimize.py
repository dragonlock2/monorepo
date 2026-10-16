import numpy as np

# TODO support diff loss functions

class SoftmaxCrossEntropyLoss:
    def forward(self, input, output_true):
        self.output_true = output_true

        # softmax
        exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.probs = exp_input / np.sum(exp_input, axis=1, keepdims=True)

        # cross entropy loss
        batch_size = input.shape[0]
        loss = -np.log(self.probs[np.arange(batch_size), output_true] + 1e-10)
        return np.sum(loss) / batch_size

    def backward(self):
        del_output = self.probs.copy()
        del_output[np.arange(del_output.shape[0]), self.output_true] -= 1
        return del_output

def optimize(model, input, output, loss_fn=SoftmaxCrossEntropyLoss, cfg={}, batch_size=64, epochs=10):
    loss_state = loss_fn()
    cfg["step"] = 1
    for epoch in range(epochs):
        for batch_idx in range(0, input.shape[0], batch_size):
            input_batch = input[batch_idx:batch_idx+batch_size,]
            output_batch = output[batch_idx:batch_idx+batch_size,]

            model_out = model.forward(input_batch, train=True)
            loss = loss_state.forward(model_out, output_batch)
            del_model_out = loss_state.backward()
            model.backward(del_model_out)
            model.update(cfg)
            cfg["step"] += 1
        print(f"epoch {epoch} loss: {loss:.5f}")

def MBGD(model, input, output, alpha=0.1, **kwargs):
    optimize(model, input, output, **kwargs, cfg={
        "type": "grad",
        "alpha": alpha,
    })

def Adam(model, input, output, alpha=0.001, beta1=0.9, beta2=0.999, **kwargs):
    optimize(model, input, output, **kwargs, cfg={
        "type": "adam",
        "alpha": alpha,
        "beta1": beta1,
        "beta2": beta2,
    })
