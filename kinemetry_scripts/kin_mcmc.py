import matplotlib.pyplot as plt
import numpy as np

def mcmc_full_run(eccano, mom, momerr, fitparams, priors, nwalkers, niter):

    # Define the initial theta array
    theta = np.array([ fitparams['pa'], fitparams['q'], fitparams['k1'] ])

    print(theta)

    # Define stepping methodology

    # Run emcee

    # Save relevant data

    # Save the posterior distribution as a function of frequency

    # Show corner plot for reference

    # Return
    return 0
