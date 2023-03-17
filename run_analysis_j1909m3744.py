import pickle
import os
from datetime import datetime

import corner
import numpy as np
from enterprise.signals.parameter import Uniform
from matplotlib import pyplot as plt
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from analysis import get_deltap_max, get_pta, read_data


def main():
    prefix = "J1909-3744_NANOGrav_12yv4"
    parfile = f"{prefix}.gls.par"
    timfile = f"{prefix}.tim"
    psr, noise_dict = read_data(
        parfile,
        timfile,
        prefix,
        noise_dict_file="channelized_12p5yr_v3_full_noisedict.json",
    )

    name = "gwecc"

    tref = max(psr.toas)
    deltap_max = get_deltap_max(psr)
    priors = {
        "sigma": Uniform(0, np.pi)(f"{name}_sigma"),
        "rho": Uniform(-np.pi, np.pi)(f"{name}_rho"),
        "log10_M": 9.0,  # Uniform(6, 9)(f"{name}_log10_M"),
        "eta": 0.25,  # Uniform(0, 0.25)(f"{name}_eta"),
        "log10_F": -8,  # Uniform(-9, -7)(f"{name}_log10_F"),
        "e0": 0.5,  # Uniform(0.01, 0.8)(f"{name}_e0"),
        "l0": 0.0,  # Uniform(-np.pi, np.pi)(f"{name}_l0"),
        "tref": tref,
        "log10_A": Uniform(-11, -5)(f"{name}_log10_A"),
        "deltap": Uniform(0, deltap_max),
        "psrTerm": False,
        "spline": True,
    }

    pta = get_pta(psr, noise_dict, prior_dict=priors)
    print("Free parameters :", pta.param_names)

    # Make sure that the PTA object works.
    # This also triggers JIT compilation of julia code and the caching of ENTERPRISE computations.
    x0 = np.array([p.sample() for p in pta.params])
    print("Log-likelihood at", x0, "is", pta.get_lnlikelihood(x0))

    outdir = "chains_" + datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss") + "/"
    if not os.path.exists(outdir):
        print("Creating output dir...")
        os.mkdir(outdir)

    summary = {
        "parfile": parfile,
        "timfile": timfile,
        "noise_dict": noise_dict,
        # "gwecc_params": priors,
    }
    with open(f"{outdir}/summary.pkl", "wb") as summarypkl:
        print("Pickling summary...")
        pickle.dump(summary, summarypkl)

    description = "Test run"
    with open(f"{outdir}/desc.txt", "w") as descfile:
        print("Writing description file...")
        descfile.write(description)

    ndim = len(x0)
    cov = np.diag(np.ones(ndim) * 0.01**2)
    Niter = 1000000
    x0 = np.hstack(x0)
    print("Starting sampler...\n")
    sampler = ptmcmc(
        ndim,
        pta.get_lnlikelihood,
        pta.get_lnprior,
        cov,
        outDir=outdir,
        resume=False,
        verbose=True,
    )
    # This sometimes fails if the acor package is installed, but works otherwise.
    # I don't know why.
    sampler.sample(x0, Niter)

    print("")

    chain_file = f"{outdir}/chain_1.txt"
    chain = np.loadtxt(chain_file)
    print("Chain shape :", chain.shape)

    burn = chain.shape[0] // 4
    burned_chain = chain[burn:, :-4]

    print("Savving plots...")
    for i in range(ndim):
        plt.subplot(ndim, 1, i + 1)
        param_name = pta.param_names[i]
        plt.plot(burned_chain[:, i])
        # plt.axhline(true_params[param_name], c="k")
        plt.ylabel(param_name)
    plt.savefig(f"{outdir}/chains.pdf")

    corner.corner(burned_chain, labels=pta.param_names)
    plt.savefig(f"{outdir}/corner.pdf")

    print("Done.")


if __name__ == "__main__":
    main()
