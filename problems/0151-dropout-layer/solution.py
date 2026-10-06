import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer.
        
        Attributes to set:
            self.p: the dropout rate
            self.mask: stores the dropout mask (initially None)
        """
        self.p = p
        self.mask = None
        self.p_inv = 1 - p
        self.rng = np.random.RandomState(seed=42)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of the dropout layer.
        
        Generate a new mask on each training forward pass and store it in self.mask.
        """
        if training:
            # Generate Bernoulli samples (0 or 1) and convert to boolean
            self.mask = self.rng.binomial(n=1, p=self.p_inv, size=x.shape)
            return (x * self.mask) / self.p_inv
        else:
            return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the dropout layer.
        
        Use the stored self.mask from the most recent forward pass.
        """
        return (grad * self.mask) / self.p_inv