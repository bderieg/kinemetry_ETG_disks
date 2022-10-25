import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from plotbin.plot_velfield import plot_velfield
import pandas as pd
import time
from astropy.io import fits

from kinemetry_scripts.kinemetry import kinemetry
import kinemetry_scripts.kinemetry as kin

import plotting_scripts as plotter

#############################
# Define user-set variables #
#############################

sb_filename = './examples/NGC4473r.fits'
vel_filename = './examples/NGC2974_SAURON_kinematics.dat'

###################
# Import map data #
###################

num, xbin, ybin, velbin, er_velbin, sigbin, er_sigbin = np.genfromtxt(vel_filename, unpack=True)

sb_img = fits.open(sb_filename)[0].data

##############################
# Do the main kinemetry task #
##############################

k = kin.kinemetry(xbin, ybin, velbin)

sbk = kin.kinemetry(img=sb_img, x0=450, y0=450, error=np.sqrt(sb_img), paq=[88,1-0.43], allterms=False, even=True, ntrm=10, bmodel=True, plot=True, nogrid=True, fixcen=False)

fig1 = plotter.plot_kinemetry_profiles(k)
fig2 = plotter.plot_vlos_maps(xbin, ybin, velbin, k, sigma=False)
fig3 = plotter.plot_sb_maps(sb_img, sbk)
plt.show()
