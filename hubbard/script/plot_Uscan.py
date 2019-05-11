import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import rcParams

import check

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
U_list = [1, 4, 8]

## Ideally implement something that will save all of this information in a dataframe for increased accessibility

# dict_keys(['nsite', 'nocc', 'sigma', 'U', 'e', 'hd', 'pair', 'pop', 'rdm1', 'nsample'])
# namelist = ["e", "hd", "pair", "pop", "rdm1"]


## Load data for several sets of job parameters, and see how well we can fit on-site terms from neighboring populations
for U in U_list:
    data = check.read_data_Hubbard(parent_path, 10, 5, 0.1, U) # (parent_path, nsite, nocc, sigma, U)
    pairs = data['pair'] # np array, 100x10, containing on-site pair densities for all 10 sites across 100 Hubb-Anderson lattices
    pops = data['pop'] # np array, 100x10, containing 1-pdm diagonal populations, 1pdm analogy of pairs

    ## Plot pair vs pop over a range of input parameters
    plt.scatter(pops.flatten(), pairs.flatten(), label=str(U))

plt.legend()
plt.ylabel("2-pdm Diagonal Elements")
plt.xlabel("1-pdm Diagonal Elements")
#plt.show()
plt.savefig("n10_no5_s01_Uscan_1pdmV2pdm.png", transparent=True, bbox_inches="tight", pad_inches=0.05)


#plt.savefig("irpcet2_gccTafel.png", transparent=True, bbox_inches='tight', pad_inches=0.05)






## Select feature vector of adjacent pops for each site, and fit to pair densities









