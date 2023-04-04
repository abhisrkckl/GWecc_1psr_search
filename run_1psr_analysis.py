import json
import os
import sys
import platform
from datetime import datetime

import corner
import numpy as np
import pickle
from dynesty import DynamicNestedSampler
from dynesty.results import print_fn as dynesty_print_fn
from enterprise.signals.parameter import Uniform
from matplotlib import pyplot as plt
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from analysis import get_deltap_max, get_pta, read_data, prior_transform_fn


def main():
    settings_file = sys.argv[1]
    with open(settings_file, "r") as sf:
        settings = json.load(sf)

    if "--no-sampler" in sys.argv:
        settings["run_sampler"] = False

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
    ecw_params = {
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
    ecw_params.update(settings["ecw_frozen_params"])

    # vary_red_noise = settings["vary_red_noise"]
    pta = get_pta(
        psr,
        noise_dict,
        ecw_params,
        noise_only=settings["noise_only"],
        wn_vary=settings["white_noise_vary"],
        rn_vary=settings["red_noise_vary"],
        rn_components=settings["red_noise_nharms"],
    )
    print("Free parameters :", pta.param_names)

    # Make sure that the PTA object works.
    # This also triggers JIT compilation of julia code and the caching of ENTERPRISE computations.
    print("Testing likelihood and prior...")
    x0 = np.array([p.sample() for p in pta.params])
    print("Log-prior at the test point is", pta.get_lnprior(x0))
    print("Log-likelihood at the test point is", pta.get_lnlikelihood(x0))

    output_prefix = settings["output_prefix"]
    jobid = "" if "SLURM_JOB_ID" not in os.environ else os.environ["SLURM_JOB_ID"]
    time_suffix = datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
    outdir = f"{output_prefix}_{jobid}_{time_suffix}/"
    if not os.path.exists(outdir):
        print(f"Creating output dir {outdir}...")
        os.mkdir(outdir)

    summary = (
        {
            "user": os.environ["USER"],
            "os": platform.platform(),
            "machine": platform.node(),
            "slurm_job_id": os.environ["SLURM_JOB_ID"]
            if "SLURM_JOB_ID" in os.environ
            else "",
        }
        | settings
        | {
            "output_dir": outdir,
            "pta_param_summary": get_pta_param_summary(pta),
        }
    )
    with open(f"{outdir}/summary.json", "w") as summarypkl:
        print("Saving summary...")
        json.dump(summary, summarypkl, indent=4)

    if settings["run_sampler"]:

        if settings["sampler"] == "ptmcmc":
            run_ptmcmc(pta, settings["ptmcmc_niter"], outdir)
            burned_chain = read_ptmcmc_chain(outdir, settings["ptmcmc_burnin_fraction"])
        elif settings["sampler"] == "dynesty":
            burned_chain = run_dynesty(pta, outdir)

        print("")

        if settings["create_plots"]:
            print("Saving plots...")

            ndim = burned_chain.shape[1]

            for i in range(ndim):
                plt.subplot(ndim, 1, i + 1)
                param_name = pta.param_names[i]
                plt.plot(burned_chain[:, i])
                # plt.axhline(true_params[param_name], c="k")
                plt.ylabel(param_name)
            plt.savefig(f"{outdir}/chains.pdf")

            corner.corner(burned_chain, labels=pta.param_names)
            plt.savefig(f"{outdir}/corner.pdf")

            if not settings["noise_only"]:
                plt.clf()
                plot_upper_limit(
                    pta.param_names,
                    burned_chain,
                    "gwecc_log10_F",
                    "$\\log_{10} f_{gw}$ (Hz)",
                    (-9, -7),
                    xparam_bins=16,
                    quantile=0.95,
                )
                plt.savefig(f"{outdir}/upper_limit_freq.pdf")

                plt.clf()
                plot_upper_limit(
                    pta.param_names,
                    burned_chain,
                    "gwecc_log10_M",
                    "$\\log_{10} M$ (Msun)",
                    (6, 9),
                    xparam_bins=8,
                    quantile=0.95,
                )
                plt.savefig(f"{outdir}/upper_limit_mass.pdf")

                plt.clf()
                plot_upper_limit(
                    pta.param_names,
                    burned_chain,
                    "gwecc_e0",
                    "$e_0$",
                    (0.01, 0.8),
                    xparam_bins=8,
                    quantile=0.95,
                )
                plt.savefig(f"{outdir}/upper_limit_ecc.pdf")

    print("Done.")


def get_pta_param_summary(pta):
    def get_prior_summary(param):
        return {"type": param.type, "kwargs": param.prior.func_kwargs}

    return {par.name: get_prior_summary(par) for par in pta.params}


def plot_upper_limit(
    param_names,
    chain,
    xparam_name,
    xparam_label,
    xparam_lims,
    xparam_bins=16,
    quantile=0.95,
):
    ampl_idx = param_names.index("gwecc_log10_A")
    freq_idx = param_names.index(xparam_name)

    chain_ampl = chain[:, ampl_idx]
    chain_freq = chain[:, freq_idx]

    log10_F_min, log10_F_max = xparam_lims
    log10_F_lins = np.linspace(log10_F_min, log10_F_max, xparam_bins + 1)
    log10_F_mins = log10_F_lins[:-1]
    log10_F_maxs = log10_F_lins[1:]
    log10_F_mids = (log10_F_mins + log10_F_maxs) / 2

    log10_A_quant_Fbin = []
    for log10_fmin, log10_fmax in zip(log10_F_mins, log10_F_maxs):
        freq_mask = np.logical_and(chain_freq >= log10_fmin, chain_freq < log10_fmax)
        log10_A_Fbin_data = chain_ampl[freq_mask]
        log10_A_quant = np.quantile(log10_A_Fbin_data, quantile)
        log10_A_quant_Fbin.append(log10_A_quant)

    log10_A_quant = np.quantile(chain_ampl, quantile)

    plt.plot(log10_F_mids, log10_A_quant_Fbin)
    plt.axhline(log10_A_quant, color="grey")
    plt.xlabel(xparam_label)
    plt.ylabel(f"{int(100*quantile)}%" + " upper bound on $\\log_{10} \\zeta_0$ (s)")


def run_ptmcmc(pta, Niter, outdir):
    x0 = np.array([p.sample() for p in pta.params])
    ndim = len(x0)
    cov = np.diag(np.ones(ndim) * 0.01**2)
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


def read_ptmcmc_chain(outdir, burnin_fraction):
    chain_file = f"{outdir}/chain_1.txt"
    chain = np.loadtxt(chain_file)
    print("Chain shape :", chain.shape)

    # burnin_fraction = settings["ptmcmc_burnin_fraction"]
    burn = int(chain.shape[0] * burnin_fraction)
    return chain[burn:, :-4]

def print_dynesty_progress(
    results,
    niter,
    ncall,
    **kwargs,
):
    if niter % 1000 == 0:
        dynesty_print_fn(results, niter, ncall, **kwargs)

def run_dynesty(pta, outdir):
    prior_transform = prior_transform_fn(pta)
    ndim = len(pta.param_names)
    sampler = DynamicNestedSampler(pta.get_lnlikelihood, prior_transform, ndim, nlive=1000)
    sampler.run_nested(dlogz=0.1, print_progress=True, print_func=print_dynesty_progress)
    res = sampler.results

    result_pkl = f"{outdir}/dynesty_result.pkl"
    with open(result_pkl, "wb") as respkl:
        pickle.dump(respkl)
    
    return res.samples_equal()


if __name__ == "__main__":
    main()
