# Network Control Theory (NCT) Modeling of Dynamics

[] set up diff eq problem using a linear model (do both discrete and continuous)

# Standard NCT Linear Model Tests
	[] build NCT diff eq function
		[] continuous
		[] discrete
	[] Constraint functions:
		[] Aij > 0 for all ij
		[] sum(Ai over j) = 1 for all i ? -> this needs to constrain dynamics to not explode
		[] sparsity constraints?


	# Resources:
	https://www.reddit.com/r/MachineLearning/comments/jgz6g2/d_best_repository_for_neural_odes/
	https://www.kaggle.com/code/shivanshuman/learning-physics-with-pytorch


# Nonlinear models:
	[] add a nonlinear dynamics model?


# Probabilistic models:
	[] Does A even make sense?
	[] Assumptions brain region i has connections to other regions
	[] Change in state of region i is dependent on:
		- Current state of i
		- Current states of j ¥ some subset of connected regions
		- Probability of brain content affecting region i?
	[] This is just a hidden markov model is it not?
