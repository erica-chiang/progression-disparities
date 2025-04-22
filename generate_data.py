import os
import numpy as np
import argparse
import pickle
import math
import random
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="test_data", help="Name of directory to save generated data in")
    parser.add_argument('--start_id', type=int, default=0, help="Index of first dataset to generate")
    parser.add_argument('--num_datasets', type=int, default=1, help="Number of datasets to save")
    parser.add_argument('--N', type=int, default=1000, help="Number of individuals to generate data for in each dataset")
    parser.add_argument('--x_n', type=int, default=4, help="Number of observed features")
    parser.add_argument('--a_n', type=int, default=2, help="Number of demographic features")
    
    args = parser.parse_args()
    return args

# Create synthetic data
def trunc_normal(mu, sigma, lower):
    r = np.random.normal(mu, sigma)
    while r < lower:
        r = np.random.normal(mu, sigma)
    return r

def generate_data(N, x_n, a_n, job_num):
    num_timesteps = 1
    
    # GLOBAL PARAMETERS / LEARNED VALUES
    F = np.array([trunc_normal(1,1, 0.5),
                  np.random.normal(0,2),
                  np.random.normal(0,2),
                  np.random.normal(0,2)
                ]) 

    F_intercept = np.random.normal(size=(x_n,)) # intercept values for f
    f = lambda z : F * z + F_intercept # f(z) = F * z
    
    mu_r = np.array([np.random.normal(1,4), np.random.normal(1,4)]) # mu for r distributions
    sigma_r = np.array([trunc_normal(0.1,0.4,0), trunc_normal(0.1,0.4,0)]) # sigma for r distributions
    
    mu_z0 = np.array([0, np.random.normal(0,4)]) # mu for z0 distributions
    sigma_z0 = np.array([1, trunc_normal(1,0.1,0)]) # sigma for z0 distributions
    
    sigma_eps = np.array([trunc_normal(5,1,0), trunc_normal(5,1,0), trunc_normal(5,1,0), trunc_normal(5,1,0)])

    beta_0 = np.random.normal(1.5, 0.1)
    beta_a = np.random.normal(0, 2)
    beta_z = trunc_normal(0.5, 0.1, 0.1)
        
    lam = lambda a, z : math.exp(np.dot(np.array([0, beta_a]), a) + beta_z * z + beta_0)

    p_x = np.random.uniform(0.7, 1, x_n) # probability of observing feature given a visit
    
    latent_params = {'mu_r':mu_r, 'sigma_r':sigma_r, 'mu_z0':mu_z0, 'sigma_z0':sigma_z0, 'r':[], 'z0':[], 'z':[], 'F':F, 
                     'F_intercept':F_intercept, 'beta_a':beta_a, 'beta_z':beta_z, 'beta_0':beta_0, 'sigma_eps':sigma_eps}

    # INDIVIDUAL DATA
    N_obs = 0
    observed_data = {'patient_id':[], 'a_per_patient':[], 'a_per_visit':[], 'X':[], 't':[], 'num_timesteps':num_timesteps, 
                    'N':N, 'x_n':x_n, 'x_n_constrained':1, 'a_n':a_n, 'num_patient_visits':[], 'patient_visit_timestep_sum':[],
                    'num_measurements':np.array([0,0,0,0]), 'measurement_times':[[],[],[],[]]}

    for patient in range(1, N+1): 
        a_indices = np.random.binomial(1, 0.5, size=(1,)) # demographic variable
        a = np.array([1-a_indices[0], a_indices[0]]) # 1-hot vector encoding of a
        num_patient_visits = 0
        
        r = np.random.normal(np.dot(mu_r, a), np.dot(sigma_r, a)) # rate of progression after first visit
        z0 = np.random.normal(np.dot(mu_z0, a), np.dot(sigma_z0, a)) # severity at time 0
        
        latent_params['z0'].append(z0)
        latent_params['r'].append(r)
        observed_data['a_per_patient'].append(a)
        
        t = 0
        num_patient_visits = 0
        sum_timesteps = 0
        z_max = max(z0, r * num_timesteps + z0)
        lambda_max = lam(a, z_max)
        
        while t <= num_timesteps: 
            z = r * t + z0 # disease severity, scalar
            epsilon = np.array([np.random.normal(scale=sigma_eps[0]),
                                np.random.normal(scale=sigma_eps[1]),
                                np.random.normal(scale=sigma_eps[2]),
                                np.random.normal(scale=sigma_eps[3])
                                ])
            
            x_all = f(z) + epsilon # symptoms, vector of x_n values

            if (t == 0) or (random.random() <= (lam(a, z) / lambda_max)): # observe event at t
                measurements = np.random.binomial(1, p_x) # binary array indicating which features were measured
                observed_data['num_measurements'] += measurements 
                for i in range(x_n):
                    if measurements[i]: observed_data['measurement_times'][i].append(len(observed_data['t']) + 1) # add index of datapoint to observe; add 1 because stan is 1-indexed
                measurements = measurements.astype('float')
                measurements[measurements == 0] = np.nan
                x = x_all * measurements

                observed_data['patient_id'].append(patient)
                observed_data['a_per_visit'].append(a)
                observed_data['X'].append(x)
                observed_data['t'].append(t)
                latent_params['z'].append(z)
                sum_timesteps += t
                num_patient_visits += 1
            u = random.random()
            t += -1.0/lambda_max * math.log(1 - u)
            
        observed_data['num_patient_visits'].append(num_patient_visits)
        observed_data['patient_visit_timestep_sum'].append(sum_timesteps)
        N_obs += num_patient_visits
            
    observed_data['N_obs'] = N_obs
    
    observed_data['a_per_visit'] = np.array(observed_data['a_per_visit'])
    observed_data['X'] = np.array(observed_data['X'])
    observed_data['t'] = np.array(observed_data['t'])
    for i in range(x_n):
        while(len(observed_data['measurement_times'][i]) < N_obs):
            observed_data['measurement_times'][i].append(0)
            
    assert(np.count_nonzero(~np.isnan(observed_data['X'][:,0])) == observed_data['num_measurements'][0])
    assert(np.count_nonzero(~np.isnan(observed_data['X'][:,1])) == observed_data['num_measurements'][1])
    assert(np.count_nonzero(~np.isnan(observed_data['X'][:,2])) == observed_data['num_measurements'][2])
    assert(np.count_nonzero(~np.isnan(observed_data['X'][:,3])) == observed_data['num_measurements'][3])
    
    return {'observed_data':observed_data,
            'latent_params':latent_params
           }

def main():
    args = get_args()

    if not os.path.exists("generated_data"):
        os.makedirs("generated_data")
    if not os.path.exists(os.path.join("generated_data", args.data_dir)):
        os.mkdir(os.path.join("generated_data", args.data_dir))

    tqdm.write("Dataset ID\tTotal # visits")
    
    for i in tqdm(range(args.start_id, args.num_datasets), dynamic_ncols=True):
        simulated_data = generate_data(N = args.N, 
                                       x_n = args.x_n,
                                       a_n = args.a_n,
                                       job_num = i)
        
        tqdm.write(str(i) + "\t\t" + str(simulated_data['observed_data']['N_obs']))

        file = open(os.path.join("generated_data", args.data_dir, 'visits_{}.pkl'.format(i)), 'wb')
        pickle.dump(simulated_data, file)
        file.close()
    
if __name__ == "__main__":
    main()