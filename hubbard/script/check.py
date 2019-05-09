"""
Check if the 1PDMs and pair densities generated match
the correponding energies.

This script also illustrates how to read those data files.
"""


import sys
import numpy as np


def get_Hubbard(n, U, nocc=None, pbc=None):
    """Generate 1D Hubbard model Hamiltonian.
    """
    if pbc is None and nocc is None:
        raise ValueError("Either arg 3 (nocc) or arg 4 (pbc) should be given.")
    elif pbc is None:
        pbc = True if nocc%2 else False

    h = np.zeros([n, n])
    for i in range(n-1):
        h[i, i+1] = h[i+1, i] = -1.
    h[0, n-1] = h[n-1, 0] = -1. if pbc else +1.

    V = np.zeros([n,n,n,n])
    for i in range(n):
        V[i,i,i,i] = U

    return h, V


def read_data_Hubbard(parent_path, nsite, nocc, sigma, U):
    """Read all relevant data for a Hubbard calculation from path

    Inp:
        parent_path (str):
            directory where the data directory is placed
        nsite (int):
            # of sites
        nocc (int):
            occupation #
        sigma (float):
            standard deviation of the Gaussina distribution from which
            the diagonal of h is sampled
        U (float):
            Hubbard U

    Return:
        data (dict):
            a dictionary containing all relevant information (see below).

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
    data = {"nsite": nsite, "nocc": nocc, "sigma": sigma, "U": U}
    path = "{:s}/n{:d}_no{:d}_s{:.1f}_U{:d}".format(\
        parent_path, nsite, nocc, sigma, U)
    namelist = ["e", "hd", "pair", "pop", "rdm1"]
    for name in namelist:
        data[name] = np.loadtxt("{:s}/{:s}_list".format(path,name))
    data["nsample"] = data["e"].shape[0]

    return data


def sanity_check_Hubbard(data):
    """Some sanity checks for the data
    """
    nsample, nsite, nocc, U = data["nsample"], data["nsite"], \
        data["nocc"], data["U"]
    # all np.arrays have the correct dimension
    assert(data["e"].shape == (nsample, ))
    assert(data["hd"].shape == (nsample, nsite))
    assert(data["pair"].shape == (nsample, nsite))
    assert(data["pop"].shape == (nsample, nsite))
    assert(data["rdm1"].shape == (nsample*nsite, nsite))

    e_recompute_list = np.zeros_like(data["e"])
    for i in range(nsample):
        rdm1 = data["rdm1"][i*nsite:(i+1)*nsite,:]
        pop = data["pop"][i,:]
        # pop is indeed diagonal of rdm1
        assert(np.allclose(np.diag(rdm1), pop))

        pair = data["pair"][i,:]
        hd = data["hd"][i,:]
        h, V = get_Hubbard(nsite, U, nocc=nocc) 
        h += np.diag(hd)
        e = data["e"][i]
        e_recompute_list[i] = 2.*np.sum(h*rdm1) + np.sum(U*pair)
    # check energy is correct
    assert(np.allclose(data["e"], e_recompute_list))


def check_all():
    parent_path = "../data/"
    nsite_list = [10, 12]
    sigma_list = [0.1, 0.3, 0.5, 1.0]
    U_list = [1, 4, 8]
    for nsite in nsite_list:
        for nocc in range(1,nsite//2+1):
            for sigma in sigma_list:
                for U in U_list:
                    print("{:d} {:d} {:.1f} {:d}".format(nsite, nocc, sigma, U))
                    data = read_data_Hubbard(parent_path, nsite, \
                        nocc, sigma, U)
                    sanity_check_Hubbard(data)


if __name__ == "__main__":
    check_all()
