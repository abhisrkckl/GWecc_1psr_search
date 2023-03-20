import json
import os
import pickle
import sys
from datetime import datetime

import corner
import numpy as np
from enterprise.signals.parameter import Uniform
from matplotlib import pyplot as plt
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from analysis import get_deltap_max, get_pta, read_data


def main():
    settings_file = sys.argv[1]
    with open(settings_file, "r") as sf:
        settings = json.load(sf)

    data_dir = settings["data_dir"]
    par_file = settings["par_file"]
    tim_file = settings["tim_file"]
    noise_file = settings["noise_dict_file"]

    psr, noise_dict = read_data(
        data_dir,
        par_file,
        tim_file,
        noise_file,
    )

    name = settings["ecw_name"]

    tref = max(psr.toas)
    deltap_max = get_deltap_max(psr)
    priors = {
        "sigma": Uniform(0, np.pi)(f"{name}_sigma"),
        "rho": Uniform(-np.pi, np.pi)(f"{name}_rho"),
        "log10_M": Uniform(6, 9)(f"{name}_log10_M"),
        "eta": Uniform(0, 0.25)(f"{name}_eta"),
        "log10_F": Uniform(-9, -7)(f"{name}_log10_F"),
        "e0": Uniform(0.01, 0.8)(f"{name}_e0"),
        "l0": Uniform(-np.pi, np.pi)(f"{name}_l0"),
        "tref": tref,
        "log10_A": Uniform(-11, -5)(f"{name}_log10_A"),
        "deltap": Uniform(0, deltap_max),
        "psrTerm": settings["ecw_psrTerm"],
        "spline": settings["ecw_spline"],
    }
    priors.update(settings["ecw_frozen_params"])

    vary_red_noise = settings["vary_red_noise"]
    pta = get_pta(psr, vary_red_noise, noise_dict, priors)
    print("Free parameters :", pta.param_names)

    # Make sure that the PTA object works.
    # This also triggers JIT compilation of julia code and the caching of ENTERPRISE computations.
    x0 = np.array([p.sample() for p in pta.params])
    print("Log-likelihood at", x0, "is", pta.get_lnlikelihood(x0))

    output_prefix = settings["output_prefix"]
    time_suffix = datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
    outdir = f"{output_prefix}_{time_suffix}/"
    if not os.path.exists(outdir):
        print(f"Creating output dir {outdir}...")
        os.mkdir(outdir)

    summary = settings | {"output_dir": outdir, "pta_free_params": pta.param_names}
    with open(f"{outdir}/summary.json", "w") as summarypkl:
        print("Pickling summary...")
        json.dump(summary, summarypkl, indent=4)

    ndim = len(x0)
    cov = np.diag(np.ones(ndim) * 0.01**2)
    Niter = settings["ptmcmc_niter"]
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

    burnin_fraction = settings["ptmcmc_burnin_fraction"]
    burn = int(chain.shape[0] * burnin_fraction)
    burned_chain = chain[burn:, :-4]

    print("Saving plots...")
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
