import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pickle
from sklearn.preprocessing import StandardScaler

def logistic(t, K, r, t0):
    """
    Logistic function:
    C(t) = K / [1 + exp(-r * (t - t0))]
    """
    return K / (1.0 + pm.math.exp(-r * (t - t0)))

def bayesian_logistic_fit(t_data, conc_data,
                          K_prior=None, r_prior=None, t0_prior=None,
                          draws=1000, tune=500, chains=2, target_accept=0.9):
    """
    Fits a Bayesian logistic model using PyMC3.
    
    Parameters
    ----------
    t_data      : 1D array of time points
    conc_data   : 1D array of observed concentrations
    K_prior     : (mu, sigma) for LogNormal prior on K  (optional)
    r_prior     : (mu, sigma) for Normal prior on r      (optional)
    t0_prior    : (mu, sigma) for Normal prior on t0     (optional)
    draws       : number of MCMC draws
    tune        : number of tuning (warmup) steps
    chains      : number of MCMC chains
    target_accept : NUTS target acceptance rate
    
    Returns
    -------
    trace : PyMC trace
    model : PyMC model context
    """

    with pm.Model() as model:
        # K ~ LogNormal(...)
        #K = pm.LogNormal('K', mu=K_prior[0], sigma=K_prior[1])
        K = pm.HalfNormal('K', sigma=K_prior[0])
        # r ~ Normal(...)
        r = pm.Normal('r', mu=r_prior[0], sigma=r_prior[1])
        # t0 ~ Normal(...)
        t0 = pm.Normal('t0', mu=t0_prior[0], sigma=t0_prior[1])

        # Noise (uncertainty)
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=1.0)

        # Deterministic logistic mean
        mu = logistic(t_data, K, r, t0)

        # Likelihood
        # If concentrations vary widely or are strictly positive, consider 
        # LogNormal / Gamma. Below we do Normal for simplicity.
        y_like = pm.Normal('y_like', mu=mu, sigma=sigma_obs, observed=conc_data)

        # MCMC sampling
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=target_accept, random_seed=42, cores=8)

    return trace, model

filt_reshaped2 = pd.read_csv('example/example.csv', sep=",")

filt_reshaped2 = filt_reshaped2.dropna(subset=['Conc_1'])

filt_reshaped2['Conc_log10'] = np.log10(filt_reshaped2['Conc_1'])

filt_reshaped2.replace([np.inf, -np.inf], np.nan, inplace=True)

filt_reshaped2 = filt_reshaped2.dropna(subset=['Conc_log10'])

disease_dict = {'total': 'total', 1: 'MM', 2: 'MGUS'}
sex_dict = {'1.0' : 'M', '2.0' : 'F', np.nan : 'NA'}
race_dict = {'1.0' : 'White', '2.0' : 'Black', '4.0' : 'Asian', np.nan : 'NA'}

filt_reshaped2['disease'] = filt_reshaped2['casestat'].map(disease_dict)

results = []

# Bayesian model fitting (Group by Patient and Clone)
for idx, (group_key, group) in enumerate(filt_reshaped2.groupby(['e_2021_1015_id', 'Clone_ID'])):
    patient_id, clone_id = group_key
    age, sex, race = group['age'].astype(float).min(), sex_dict[group['sex'].unique()[0].astype(str)], race_dict[group['race7'].unique()[0].astype(str)]
    msdx = group['Baseline_Clone_Mass_Spect_Diagnosis_SK'].unique()[0]
    disease_name = group['disease'].unique()[0]
    igh_type = group['Type_heavy'].unique()[0]

    years_ori = group['Yr_before_dx'].values
    years = group['Yr_before_dx'].values * -1 ## year before diagnosis, so negative!!!!!
    #concentration = group['Conc_log10'].values
    concentration = group['Conc_1'].values

    # Take clones with at least 3 timepoints!!
    if len(years) > 2:
        # Prior setup
        #K_prior = (np.log(10.0), 2) #np.log(15.0), 0.5
        K_prior = (10, 0) # Sigma (left) only
        #r_prior = (1, 0.5)
        r_prior = (0, 1)
        # center the inflection ~ -5 years
        #t0_prior = (-2.0, 3.0)
        #t0_prior = (0, 5)
        t0_prior = (np.mean(years), 5)

        trace, model = bayesian_logistic_fit(years, concentration,
                                             K_prior=K_prior,
                                             r_prior=r_prior,
                                             t0_prior=t0_prior
                                            )

        # Save distribution and model
        # Load trace: trace = az.from_netcdf('.nc')
        az.to_netcdf(trace, f"output/{patient_id}_{clone_id}_trace.nc")

        # Create summary df
        summary = az.summary(trace, hdi_prob=0.95)
