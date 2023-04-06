import json
import os
import sys
import platform
from datetime import datetime

import corner
import numpy as np
from matplotlib import pyplot as plt
from enterprise_extensions.hypermodel import HyperModel
from enterprise_extensions.model_utils import odds_ratio

from analysis import (
    get_ecw_params,
    get_pta,
    read_data,
)
from plotting import plot_upper_limit
from sampling import create_red_noise_empirical_distr, run_dynesty, run_ptmcmc, read_ptmcmc_chain

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

    ecw_params = get_ecw_params(psr, settings)

    # vary_red_noise = settings["vary_red_noise"]
    pta0 = get_pta(
        psr,
        noise_dict,
        ecw_params,
        noise_only=True,
        wn_vary=settings["white_noise_vary"],
        rn_vary=settings["red_noise_vary"],
        rn_components=settings["red_noise_nharms"],
    )
    pta1 = get_pta(
        psr,
        noise_dict,
        ecw_params,
        noise_only=False,
        wn_vary=settings["white_noise_vary"],
        rn_vary=settings["red_noise_vary"],
        rn_components=settings["red_noise_nharms"],
    )
    hm = HyperModel({0: pta0, 1: pta1})
    print("Free parameters :", hm.param_names)

    # Make sure that the PTA object works.
    # This also triggers JIT compilation of julia code and the caching of ENTERPRISE computations.
    print("Testing likelihood and prior...")
    x0 = hm.initial_sample()
    print("Log-prior at the test point is", hm.get_lnprior(x0))
    print("Log-likelihood at the test point is", hm.get_lnlikelihood(x0))

    output_prefix = settings["output_prefix"]
    jobid = "" if "SLURM_JOB_ID" not in os.environ else os.environ["SLURM_JOB_ID"]
    time_suffix = datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
    outdir = f"{output_prefix}_{jobid}_{time_suffix}/"
    if not os.path.exists(outdir):
        print(f"Creating output dir {outdir}...")
        os.mkdir(outdir)

    summary = get_summary(hm, outdir, settings)
    with open(f"{outdir}/summary.json", "w") as summarypkl:
        print("Saving summary...")
        json.dump(summary, summarypkl, indent=4)

    if settings["run_sampler"]:
        rn_ed = create_red_noise_empirical_distr(
            psr, "data/red_noise_empdist_samples.dat"
        )
        red_noise_group = [
            hm.param_names.index(f"{psr.name}_red_noise_gamma"),
            hm.param_names.index(f"{psr.name}_red_noise_log10_A"),
        ]
        run_ptmcmc(
            hm,
            settings["ptmcmc_niter"],
            outdir,
            groups=[red_noise_group],
            empdist=[rn_ed],
        )
        burned_chain = read_ptmcmc_chain(outdir, settings["ptmcmc_burnin_fraction"])

        print("")

        if settings["create_plots"]:
            print("Saving plots...")

            ndim = burned_chain.shape[1]

            for i in range(ndim):
                plt.subplot(ndim, 1, i + 1)
                param_name = hm.param_names[i]
                plt.plot(burned_chain[:, i])
                plt.ylabel(param_name)
            plt.savefig(f"{outdir}/chains.pdf")

        nmodel_idx = hm.param_names.index("nmodel")
        nmodel_samples = burned_chain[:, nmodel_idx]
        bf, bferr = odds_ratio(nmodel_samples, models=[0, 1])
        print("nmodel index = ", nmodel_idx)
        print(f"Bayes Factor = {bf} +/- {bferr}")

    print("Done.")


def get_pta_param_summary(pta):
    def get_prior_summary(param):
        return {"type": param.type, "kwargs": param.prior.func_kwargs}

    return {par.name: get_prior_summary(par) for par in pta.params}


def get_summary(pta, outdir, settings):
    return (
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

if __name__ == "__main__":
    main()
