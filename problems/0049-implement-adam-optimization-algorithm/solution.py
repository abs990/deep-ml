import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Note: `f` is accepted only for interface parity with the PyTorch/Tinygrad
    # variants, which derive gradients from it via autograd. This version uses
    # `grad` only — the objective value `f` is never evaluated.
    # Your code here
    
    # init
    m0 = np.zeros_like(x0)
    v0 = np.zeros_like(x0)

    for t in range(1, 1 + num_iterations):
        g = grad(x0)
        m_next = (beta1 * m0) + (1 - beta1) * g
        v_next = (beta2 * v0) + (1 - beta2) * (g ** 2)
        m_corrected = m_next / (1 - beta1 ** t)
        v_corrected = v_next / (1 - beta2 ** t)

        x0 = x0 - (learning_rate * m_corrected) / (epsilon + np.sqrt(v_corrected))

        m0 = m_next
        v0 = v_next

    return x0