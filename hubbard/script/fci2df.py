import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import rcParams

import check

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
S_list = [1., 0.5, 0.3, 0.1]
U_list = [1, 4, 8]
no_list = [1,2,3,4,5]
n_list = [10,12]

# dict_keys: ['nsite', 'nocc', 'sigma', 'U', 'e', 'hd', 'pair', 'pop', 'rdm1', 'nsample']
# namelist:  ["e", "hd", "pair", "pop", "rdm1"]

"""
We want the model to only take in 1pdm elements from all of this data, and check how 2-pdm on-site is reproduced
We'll start with only trying to fit to on-site terms, but we'll probably want to collect coherences as well
"""


## Load data for several sets of job parameters, and see how well we can fit on-site terms from neighboring populations
for n in n_list:
    for U in U_list:
        for sig in S_list:
            for no in no_list:
                data = check.read_data_Hubbard(parent_path, n, no, sig, U) # (parent_path, nsite, nocc, sigma, U)
                pairs = data['pair'] # np array, 100x10, containing on-site pair densities
                pops = data['pop'] # np array, 100x10, containing 1-pdm diagonal populations, 1pdm analogy of pairs

                # save all pops next to each site
                # I can't just ravel because I need adjacency within each matrix

                X0 = pairs.flatten()


                Y = pairs.flatten() # we don't need (or necessarily want) adjacency structure saved for these terms
                
                




