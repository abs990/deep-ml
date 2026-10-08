import numpy as np

def adaptive_kl_penalized_reward(rewards, log_probs_policy, log_probs_reference, beta, kl_target, beta_update_factor=1.5):
	"""
	Compute KL-penalized shaped rewards with adaptive beta adjustment.

	Args:
		rewards: list of scalar rewards for each response
		log_probs_policy: list of lists, per-token log-probs under policy for each response
		log_probs_reference: list of lists, per-token log-probs under reference for each response
		beta: KL penalty coefficient
		kl_target: target KL divergence for adaptive adjustment
		beta_update_factor: multiplicative factor for beta adjustment (default 1.5)

	Returns:
		dict with 'per_response_kl', 'mean_kl', 'shaped_rewards', 'updated_beta'
	"""

	per_response_kl = []

	for idx, (policy, ref) in enumerate(zip(log_probs_policy, log_probs_reference)):
		# KLD
		# per_response_kl.append(np.dot(np.exp(policy), policy - ref))
		per_response_kl.append(round(sum([p - r for p, r in zip(policy, ref)]), 4))

		# reward
		rewards[idx] -= float(beta * per_response_kl[-1])
		rewards[idx] = round(rewards[idx], 4)

	# mean
	mean_kl = sum(per_response_kl)/len(per_response_kl)

	# beta beta_update_factor
	if mean_kl > kl_target * 1.5:
		beta *= beta_update_factor
	if mean_kl < kl_target / 1.5:
		beta /= beta_update_factor

	return {
		'per_response_kl': per_response_kl,
		'mean_kl': mean_kl,
		'shaped_rewards': rewards,
		'updated_beta': round(beta, 4)
	}