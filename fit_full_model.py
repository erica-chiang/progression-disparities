import numpy as np
import stan
import arviz as az
import argparse
import pickle
import os
from sklearn.decomposition import FactorAnalysis

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', type=int, default=0, help="Index of dataset to fit stan model on")
    parser.add_argument('--data_file_substr', type=str, default="visits_{}.pkl", help="Name structure of data files")
    parser.add_argument('--data_dir', type=str, default="test_data", help="Path of directory with datasets to fit stan model on")
    parser.add_argument('--stan_dir', type=str, help="Path of directory to store stan output in")

    parser.add_argument('--warmup', type=int, default=8000, help="Number of warmup iterations")
    parser.add_argument('--sampling', type=int, default=8000, help="Number of sample iterations")
    parser.add_argument('--num_chains', type=int, default=4, help="Number of chains")
    parser.add_argument('--stan_model', type=str, default="full_model", help="Name of stan model to fit on data")
    parser.add_argument('--k_constrained', type=int, default=1)
    parser.add_argument('--min_constrained_value_size', type=float, default=0.5)

    args = parser.parse_args()
    return args

def trunc_normal(mu, sigma):
    r = np.random.normal(mu, sigma)
    while r < 0:
        r = np.random.normal(mu, sigma)
    return r

def initialize_factor_analysis_from_data(X, K_constrained, n_chains, min_constrained_value_size, noise_in_initialization=0.01):
    """
    This estimates the factor loadings using sklearn's FactorAnalysis, and then initializes the Stan model with these values.
    Note this fits only A and the noise variance, doesn't fit b. 
    """

    K = len(X[0])
    fa = FactorAnalysis(n_components=1)
    fa.fit(X)
    F_initial_values = fa.components_.flatten()
    if F_initial_values[0] < 0: # flip the sign of the first component so that it's positive
        F_initial_values = -F_initial_values
        
    for k in range(K_constrained):
        assert(F_initial_values[k] > 0) # this should automatically fix the other constrained components in a well-specified model. 
    noise_std_initial_values = np.sqrt(fa.noise_variance_)
    stan_initialization = [{'F_constrained': np.clip(F_initial_values[:K_constrained] + np.random.randn(K_constrained).flatten() * noise_in_initialization, min_constrained_value_size + 0.01, None), 
                               'F_unconstrained': F_initial_values[K_constrained:] + np.random.randn(K - K_constrained).flatten() * noise_in_initialization, 
                               'sigma_eps':np.clip(noise_std_initial_values + np.random.randn(K) * noise_in_initialization, 0.01, None),
                               'F_intercept': np.random.randn(K),
                               'beta_0': np.random.normal(1.5, 0.1),
                               'beta_z': trunc_normal(0.5, 0.1) + 0.01} 
                               for _ in range(n_chains)]
    return stan_initialization

def main():
    args = get_args()
    data_file_substr_full = os.path.join("generated_data", args.data_dir, args.data_file_substr)
    data_file = data_file_substr_full.format(args.job_id)
    stan_model_full = args.stan_model + ".txt"
    stan_file = os.path.join("stan_models", stan_model_full)
    
    if args.stan_dir:
        stan_dir = args.stan_dir
    else:
        stan_dir = os.path.join("stan_output", args.data_dir + "_" + args.stan_model)
        
    if not os.path.exists("stan_output"):
        os.makedirs("stan_output")
    if not os.path.exists(stan_dir):
        os.makedirs(stan_dir)
    
    file = open(data_file, 'rb')
    simulated_data = pickle.load(file)
    file.close()

    # build and fit model
    i_with_t0 = np.where(simulated_data['observed_data']['t'] == 0)[0]
    i_with_a0 = np.where(simulated_data['observed_data']['a_per_visit'][:,0])[0]
    mask = np.isnan(simulated_data['observed_data']['X']).any(axis=1)
    i_with_allvalues = np.where(~mask)[0]

    i_for_factor_analysis = set(set(i_with_t0).intersection(i_with_a0)).intersection(i_with_allvalues)
    x_a0_t0 = [simulated_data['observed_data']['X'][i] for i in i_for_factor_analysis]

    init_values = initialize_factor_analysis_from_data(x_a0_t0, K_constrained=args.k_constrained, n_chains=args.num_chains, min_constrained_value_size=args.min_constrained_value_size, noise_in_initialization=0.05)

    # get model
    with open(stan_file) as file:
        stan_code = file.read()
    
    model = stan.build(stan_code, data=simulated_data['observed_data'])
    
    fit = model.sample(num_chains=args.num_chains, num_warmup=args.warmup, num_samples=args.sampling, init=init_values)

    # get arviz summary
    az_df = az.summary(fit)
    az_df.to_csv(os.path.join(stan_dir, "az_sum_fit_{}.csv".format(args.job_id)))
    summ = az.summary(fit)
    summ = summ.drop(columns = summ.columns.difference(['mean','sd', 'r_hat']))
    print(summ)
    
    axes = az.plot_posterior(az.convert_to_dataset(fit), 
                             var_names = ['F_constrained', 'F_unconstrained', 'F_intercept', 'beta_0', 'beta_a', 'beta_z', 'mu_r', 'sigma_r', 'mu_z0_1', 'sigma_z0_1', 'sigma_eps'], 
                             ref_val = [simulated_data['latent_params']['F'][0], simulated_data['latent_params']['F'][1], simulated_data['latent_params']['F'][2], simulated_data['latent_params']['F'][3], simulated_data['latent_params']['F_intercept'][0], simulated_data['latent_params']['F_intercept'][1], simulated_data['latent_params']['F_intercept'][2], simulated_data['latent_params']['F_intercept'][3], simulated_data['latent_params']['beta_0'], simulated_data['latent_params']['beta_a'], simulated_data['latent_params']['beta_z'], simulated_data['latent_params']['mu_r'][0], simulated_data['latent_params']['mu_r'][1], simulated_data['latent_params']['sigma_r'][0], simulated_data['latent_params']['sigma_r'][1], simulated_data['latent_params']['mu_z0'][1], simulated_data['latent_params']['sigma_z0'][1], simulated_data['latent_params']['sigma_eps'][0], simulated_data['latent_params']['sigma_eps'][1], simulated_data['latent_params']['sigma_eps'][2], simulated_data['latent_params']['sigma_eps'][3]])

    fig = axes.ravel()[0].figure
    fig.savefig('./figures/posteriors_{}.png'.format(args.job_id))
    
    
if __name__ == "__main__":
    main()
