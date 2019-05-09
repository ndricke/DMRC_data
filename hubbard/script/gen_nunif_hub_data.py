import sys
import numpy as np

from frankenstein import lattice, fci
from frankenstein.tools.lat_utils import get_Hubbard
from frankenstein.tools.io_utils import dumpMat


def print_cmd_args(nsite, nocc, U, sigma, nsample, outpath):
    print("Command-line arguments:", flush=True)
    print("  {:10s} = {:d}".format("nsite", nsite), flush=True)
    print("  {:10s} = {:d}".format("nocc", nocc), flush=True)
    print("  {:10s} = {:.2f}".format("U", U), flush=True)
    print("  {:10s} = {:.2f}".format("sigma", sigma), flush=True)
    print("  {:10s} = {:d}".format("nsample", nsample), flush=True)
    print("  {:10s} = {:s}".format("outpath", outpath), flush=True)


def get_hd_list(nsample, nsite, sigma):
    hd_list = np.random.normal(0., sigma, nsite*nsample).reshape(nsample,nsite)
    return hd_list


def run_Hubbard(nsite, nocc, U, hd):
    h, V = get_Hubbard(nsite, U, nocc)
    hh = np.einsum("ii->i", h)
    hh += hd
    del hh

    lat = lattice.LATTICE(h=h, V=V, nocc=nocc)
    mc = fci.FCI(lat)
    mc.kernel(rdm_level=1)

    pops = np.diag(mc.rdm1[0]).copy()
    pairs = np.zeros_like(hd)
    for i in range(nsite):
        pairs[i] = \
            fci.FCI.get_rdm_elements(nsite, nocc, [[i,i,i,i]], \
                mc.ci_coeff[0])[0]

    return mc.e_fci[0], mc.rdm1[0], pops, pairs


def run_Hubbard_all(nsite, nocc, U, hd_list):
    nsample = hd_list.shape[0]
    assert(hd_list.shape[1] == nsite)

    e_list = np.zeros(hd_list.shape[0])
    rdm1_list = np.zeros([nsample, nsite, nsite])
    pop_list = np.zeros_like(hd_list)
    pair_list = np.zeros_like(hd_list)
    for i in range(nsample):
        print("{:d}".format(i+1), flush=True)
        hd = hd_list[i]
        e_list[i], rdm1_list[i], pop_list[i], pair_list[i] = \
            run_Hubbard(nsite, nocc, U, hd)

    return e_list, rdm1_list, pop_list, pair_list


def convert_rdm1_list(rdm1_list):
    if rdm1_list.ndim == 3:
        nsample, nsite, c = rdm1_list.shape
        assert(nsite == c)
        return rdm1_list.reshape(nsample*nsite, nsite)
    elif rdm1_list.ndim == 2:
        ntot, nsite = rdm1_list.shape
        assert(ntot%nsite == 0)
        nsample = ntot // nsite
        return rdm1_list.reshape(nsample, nsite, nsite)
    else:
        raise RuntimeError("Input rdm1_list has wrong dimension!")


def save_all(hd_list, e_list, rdm1_list, pop_list, pair_list, outpath):
    if rdm1_list.ndim == 3:
        rdm1_list = convert_rdm1_list(rdm1_list)
    namelist = ["hd", "e", "rdm1", "pop", "pair"]
    objlist = [hd_list, e_list, rdm1_list, pop_list, pair_list]
    for name, obj in zip(namelist, objlist):
        np.savetxt("{:s}/{:s}_list".format(outpath,name), obj)


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: nsite nocc U sigma nsample outpath")
        sys.exit(1)
    nsite = int(sys.argv[1])
    nocc = int(sys.argv[2])
    U = float(sys.argv[3])
    sigma = float(sys.argv[4])
    nsample = int(sys.argv[5])
    outpath = sys.argv[6]

    np.random.seed(17)

    # print command-line args
    print_cmd_args(nsite, nocc, U, sigma, nsample, outpath)

    # generating on-site randomness
    hd_list = get_hd_list(nsample, nsite, sigma)

    # run
    e_list, rdm1_list, pop_list, pair_list = \
        run_Hubbard_all(nsite, nocc, U, hd_list)

    # save data
    save_all(hd_list, e_list, rdm1_list, pop_list, pair_list, outpath)
