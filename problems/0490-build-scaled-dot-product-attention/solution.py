import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
	"""
	Compute Scaled Dot-Product Attention.
	
	Args:
		Q: Query matrix of shape (seq_len_q, d_k)
		K: Key matrix of shape (seq_len_k, d_k)
		V: Value matrix of shape (seq_len_k, d_v)
		mask: Optional binary mask of shape (seq_len_q, seq_len_k)
	
	Returns:
		Tuple of (output, attention_weights)
	"""
	# Your code here
	seq_len_q, d_k = Q.shape

	# 1. Compute raw scores
	S = Q @ K.T

	# 2. Scale the scores
	S = S / np.sqrt(d_k)
	
	# 3. Optional mask
	if mask is not None:
		S[mask == 0] = -float('inf')
	
	# 4. Softmax
	S = S - np.max(S, axis=1)[:, None]
	S = np.exp(S)
	S = S / np.sum(S, axis=1)[:, None]

	return S @ V, S