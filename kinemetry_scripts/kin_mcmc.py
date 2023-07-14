import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner

def mcmc_full_run(mominterp, mominterp_err, fitparams, priors, nwalkers, niter):

    # Define the initial theta array
    # theta = np.array([ fitparams['pa'], fitparams['q'], fitparams['k1'] ])
    theta = np.array([ fitparams['pa'], fitparams['q'] ])

    # Define stepping methodology
    ndim = len(theta)
    p0 = [theta+1e-7*np.random.randn(ndim) for i in range(nwalkers)]

    # Run emcee
    data = (mominterp, mominterp_err, priors, fitparams)
    sampler, pos, prob, state = run_samples(p0, nwalkers, niter, ndim, lnprob, data)

    # Save relevant data
    ## Get sample data
    samples = sampler.get_chain(flat=True, discard=500)
    ## Find the most probable datum
    theta_max = samples[np.argmax(sampler.get_log_prob(flat=True, discard=500))].copy()
    ## Draw some random samples and get statistics
    draw = np.floor(np.random.uniform(0,len(samples),size=200)).astype(int)
    theta_dist = samples[draw]
    pa_spread = np.std(theta_dist[:,0],axis=0)
    pa_median = np.median(theta_dist[:,0],axis=0)
    q_spread = np.std(theta_dist[:,1],axis=0)
    q_median = np.median(theta_dist[:,1],axis=0)
    # k1_spread = np.std(theta_dist[:,2],axis=0)
    # k1_median = np.median(theta_dist[:,2],axis=0)

    # Show corner plot for reference
    # cp_labels = ['PA (deg)','Q','$k_1$ (km s$^{-1}$)']
    # # cp_labels = ['PA (deg)','Q']
    # fig = corner.corner(
    #         samples,
    #         labels=cp_labels,
    #         quantiles=[0.16,0.5,0.84],
    #         show_titles=True,
    #         plot_datapoints=True
    #         )
    # plt.show()

    # Return
    return {
            'pa_spread' : pa_spread, 
            'pa_median' : pa_median, 
            'q_spread' : q_spread, 
            'q_median' : q_median#, 
            # 'k1_spread' : k1_spread,
            # 'k1_median' : k1_median
        }


def run_samples(p0, nwalkers, niter, ndim, lnprob, data):

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    pos, prob, state = sampler.run_mcmc(p0, niter+500)

    return sampler, pos, prob, state


def lnlike(theta, mominterp, mominterp_err, fitparams):

    # Extract data along ellipse
    ang = np.radians(theta[0] - 90)
    th_list = np.linspace(0.0, 2*np.pi, num=100)
    x = fitparams['radius'] * np.cos(th_list)
    y = fitparams['radius'] * np.sin(th_list) * theta[1]
    xEll = fitparams['x0'] + x*np.cos(ang) - y*np.sin(ang)
    yEll = fitparams['y0'] + x*np.sin(ang) + y*np.cos(ang)
    momEll = mominterp(xEll, yEll)
    momEll_err = mominterp_err(xEll, yEll)

    # Calculate residuals
    # LnLike = -0.5*np.sum( ( (momEll-fitparams['a0']-theta[2]*np.cos(th_list)) / momEll_err )**2 )
    LnLike = -0.5*np.sum( ( (momEll-fitparams['a0']-fitparams['k1']*np.cos(th_list)) / momEll_err )**2 )

    return LnLike


def lnprior(theta, priors):

    prior_test = [
            theta[0] < priors['pa_uplim'],
            theta[0] > priors['pa_lolim'],
            theta[1] < priors['q_uplim'],
            theta[1] > priors['q_lolim']#,
            # theta[2] < priors['k1_uplim'],
            # theta[2] > priors['k1_lolim']
        ]

    if np.all(prior_test):
        return 0.0
    else:
        return -np.inf


def lnprob(theta, mominterp, mominterp_err, priors, fitparams):

    lp = lnprior(theta, priors)

    if not lp == 0.0:
        return -np.inf
    else:
        return lp + lnlike(theta, mominterp, mominterp_err, fitparams)
