import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from plotbin.plot_velfield import plot_velfield
import pandas as pd
import time
from astropy.io import fits
import csv
import operator

from kinemetry_scripts.kinemetry import kinemetry
import kinemetry_scripts.kinemetry as kin

import plotting_scripts as plotter

import sys

#######################################
# Define parameter file read function #
#######################################

def read_properties(filename):
    """ Reads a given properties file with each line of the format key=value.  Returns a dictionary containing the pairs.

    Keyword arguments:
        filename -- the name of the file to be read
    """
    result={ }
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter='=', escapechar='\\', quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) == 0:
                continue
            elif len(row) != 2:
                raise csv.Error("Too many fields on row with contents: "+str(row))
            try:
                row[1] = int(row[1])
            except:
                pass
            if row[1] == "True":
                row[1] = True
            elif row[1] == "False":
                row[1] = False
            result[row[0]] = row[1]
    return result


##################################
# Attempt to read parameter file #
##################################

# Get parameter file name from command line argument
if len(sys.argv) == 2:
    param_filename = sys.argv[1]
else:
    raise Exception("Incorrect number of arguments (should be 1--parameter file name)")

# Read file
params = read_properties(param_filename)

###################
# Import map data #
###################

velmap = fits.open(params['velmap_filename'])[0].data

##############################
# Do the main kinemetry task #
##############################

k = kin.kinemetry(img=velmap, x0=params['x0'], y0=params['y0'])

fig1 = plotter.plot_kinemetry_profiles(k)
# fig2 = plotter.plot_vlos_maps(xbin, ybin, velbin, k, sigma=False)
plt.show()

