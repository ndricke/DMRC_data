import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import rcParams

import check

"""
Script to plot 1pdm diagonal elements vs 2pdm diagonal elements
Overall conclusions:
n=10 and n=12 are not very different
if the large sigma values can be modeled, it should be possible to model smaller values
different values of U have curves at different places, but they follow the same shape
higher occupation values fall along a well defined 1-2 pdm curve relation, whereas lower values may be harder

The goal of this project is to reconstruct pieces of the 2pdm with the 1pdm without any knowledge of the potentials
"""

## Plotting parameters
font = {'size':18}
mpl.rc('font',**font)
rcParams['figure.figsize'] = 10,10

"""
"data" contains:
    1. nsite, nocc, sigma, U as from input
    2. nsample (int) deduced from the quantities below
    3. e [nsample]:
        total energy
    4. hd [nsample, nsite]:
        each row is a vector of diagonal of h (on-site potential fluctuation)
    5. pair [nsample, nsite]:
        each row is a vector of pair densities
    6. pop [nsample, nsite]:
        each row is a vector of populations
    7. rdm1 [nsample*nsite, nsite]:
        every nsite rows correspond to a rdm1 (whose diagonal matches the
        corresponding row of pop)
"""


parent_path = "/home/nricke/pymod/DMRC_data/hubbard/data"
S_list = [1.]
U_list = [4]
no_list = [1,2,3,4,5]

# dict_keys: ['nsite', 'nocc', 'sigma', 'U', 'e', 'hd', 'pair', 'pop', 'rdm1', 'nsample']
# namelist:  ["e", "hd", "pair", "pop", "rdm1"]

## Load data for several sets of job parameters, and see how well we can fit on-site terms from neighboring populations
for n in [10,12]:
    for U in U_list:
        for sig in S_list:
            for no in no_list:
                data = check.read_data_Hubbard(parent_path, n, no, sig, U) # (parent_path, nsite, nocc, sigma, U)
                pairs = data['pair'] # np array, 100x10, containing on-site pair densities
                pops = data['pop'] # np array, 100x10, containing 1-pdm diagonal populations, 1pdm analogy of pairs

                ## Plot pair vs pop over a range of input parameters
                plt.scatter(pops.flatten(), pairs.flatten(), label="U=%s, sig=%s, no=%s, n=%s" % (U, sig, no, n))

plt.legend()
plt.ylabel("2-pdm Diagonal Elements")
plt.xlabel("1-pdm Diagonal Elements")
#plt.show()
plt.savefig("AllScan_1pdmV2pdm.png", transparent=True, bbox_inches="tight", pad_inches=0.05)
