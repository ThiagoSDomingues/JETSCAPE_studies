### Author: OptimusThi
"""
Script created to train emulators for heavy-ion collisions observables.
"""

import numpy as np

n_params = 17 # JETSCAPE model for one collision system

# function to make emulator's predictions: mean and standard deviation
def predict_observables(model_parameters, Emulators, inverse_tf_matrix, SS):
    model_parameters = np.array(model_parameters).flatten()
    if model_parameters.shape[0] != n_params:
        raise ValueError("Input model parameters must be a n_params-dimensional array.")
    theta = model_parameters.reshape(1, -1)
    n_pc = len(Emulators)
    pc_means = []
    pc_vars = []
    for emulator in Emulators:
        mn, std = emulator.predict(theta, return_std=True)
        pc_means.append(mn.flatten()[0])
        pc_vars.append(std.flatten()[0]**2)
    pc_means = np.array(pc_means).reshape(1, -1)
    variance_matrix = np.diag(np.array(pc_vars))
    inverse_transformed_mean = pc_means @ inverse_tf_matrix[:n_pc, :] + SS.mean_.reshape(1, -1)
    A = inverse_tf_matrix[:n_pc, :]
    inverse_transformed_variance = np.einsum('ik,kl,lj->ij', A.T, variance_matrix, A)
    return inverse_transformed_mean.flatten(), inverse_transformed_variance

# Making prior predictions
np.random.seed(1) # for reproducibility
pr_predictions = []

n_pr_samples = 2000

# Looping over all prior samples: supposing a uniform prior
for params in np.random.uniform(param_min, param_max, (n_pr_samples, n_params)):        
    y_pred, cov_pred = predict_observables(params, emulators, inv_tf, scaler)
    pr_predictions.append(y_pred.flatten())
pr_predictions = np.array(pr_predictions) # shape (n_pr_samples, n_obs)
