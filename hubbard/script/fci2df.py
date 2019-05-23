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

# given a site in the 1pdm, generate indices with pbc for all pops and coherences we want to collect for that site
def get_pdm_terms(site_index, n, adj_sites=4, shift=0):
    """
    inputs:
        site_index (int): center site index
        n (int): number of sites in lattice (for enforcing periodic boundary conditions)
        adj_site (int): how many adjacent sites to collect pop and coherence values from
        shift (int): return row indices shifted by this value (when collecting from matrix w/ compound index
    returns:
        ind_list (list of ints): all site indices to collect
        shift_ind_list (list of ints): site indices shifted down i*n rows to collect the correct sample
        coh_list (list of ints): coherences in fragment with central site
        shift_coh_list (list of ints): coherences shifted down i*n rows to select particular sample
    """

    # if a term in sites is out of bounds, subtract bound length
    ind_list, shift_ind_list = [site_index], [site_index+shift]
    coh_list, shift_coh_list = [], []
    #print("Site: ", site_index)
    for ind in range(site_index - adj_sites, site_index + adj_sites + 1):
        if ind != site_index: # we've already add the target site population to ind_list and shift_ind_list
            if ind < 0: ind += n
            elif ind >= n: ind -= n
            else: pass
            #print(ind)
            ind_list.append(ind)
            shift_ind_list.append(ind+shift) # shift down to specific matrix in set we are selecting
            coh_list.append(ind)
            shift_coh_list.append(ind+shift)

    return ind_list, shift_ind_list, coh_list, shift_coh_list


parent_path = "/home/nricke/pymod/DMRC_data/hubbard/data"
S_list = [1., 0.5, 0.3, 0.1]
U_list = [1, 4, 8]
no_list = [1,2,3,4,5]
n_list = [10,12]

adj_site_num = 4
nsamples = 100

# dict_keys: ['nsite', 'nocc', 'sigma', 'U', 'e', 'hd', 'pair', 'pop', 'rdm1', 'nsample']
# namelist:  ["e", "hd", "pair", "pop", "rdm1"]

"""
We want the model to only take in 1pdm elements from all of this data, and check how 2-pdm on-site is reproduced
We'll start with only trying to fit to on-site terms, but we'll probably want to collect coherences as well

How should I actually store this data? In one dataframe for each folder in data? Probably all of this should fit in memory, and
we can load and parse based on the values of S, U, no, and n, so a single dataframe is probably better

"""

## create column names for dataframe
column_name_list = ["n", "U", "sigma", "n_occ", "site_pop"]
L_site_list, R_site_list, L_coh_list, R_coh_list = [], [], [], []
for i in range(adj_site_num):
    L_site_list.append("L"+str(i+1)) # sites left of target set
    R_site_list.append("R"+str(i+1)) # ditto for the right
L_coh_list = ["c"+name for name in L_site_list]
R_coh_list = ["c"+name for name in R_site_list]
column_name_list = column_name_list + L_site_list + R_site_list + L_coh_list + R_coh_list + ["pair_density"]

## Load data for several sets of job parameters, and see how well we can fit on-site terms from neighboring populations
XY_df_list = []
for n in n_list:
    x = np.zeros((n*nsamples, 2*adj_site_num + 1))
    x_coh = np.zeros((n*nsamples, 2*adj_site_num))
    n_arr = np.ones((n*nsamples,1))*n
    for U in U_list:
        print("U progress: ", U)
        U_arr = np.ones((n*nsamples,1))*U
        for sig in S_list:
            sig_arr = np.ones((n*nsamples,1))*sig
            for no in no_list:
                x *= 0.
                x_coh *= 0.
                no_arr = np.ones((n*nsamples,1))*no
                data = check.read_data_Hubbard(parent_path, n, no, sig, U) # (parent_path, nsite, nocc, sigma, U)
                pairs = data['pair'] # np array, 100x10, containing on-site pair densities
                #pops = data['pop'] # np array, 100x10, containing 1-pdm diagonal populations, 1pdm analogy of pairs
                rdm1 = data['rdm1']

                # need to collect data from rdm1 if we want coherence info (which we only use if necessary)
                for i in range(nsamples): # there are 100 samples, as per the row dimension of pops and pairs
                    for j in range(n): #for each site in the system
                        ind_list, s_ind, coh_list, s_coh = get_pdm_terms(j, n, shift=i*n) # list of sites within 1pdm
                        #print(len(ind_list), len(s_ind), len(coh_list), len(s_coh))
                        x[i*n+j,:] = rdm1[s_ind, ind_list] 
                        x_coh[i*n+j,:] = rdm1[s_coh, coh_list]

                Y = pairs.flatten().reshape((-1,1))
                # merge x, x_coh, and Y into a single numpy array, along with the data for the specific calculation
                print([np.shape(arr) for arr in [n_arr, U_arr, sig_arr, no_arr, x, x_coh, Y]])
                XY = np.concatenate([n_arr, U_arr, sig_arr, no_arr, x, x_coh, Y], axis=1)
                XY_df = pd.DataFrame(XY, columns = column_name_list)  # construct df from np arrays x, x_coh, and Y
                # may be best to just save a list of dataframes, which can be concatenated at the end
                XY_df_list.append(XY_df)


print(len(XY_df_list))
df = pd.concat(XY_df_list)
df.to_csv("dmrc_data.csv")

                




