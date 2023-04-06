import json
import os
import pickle

import astropy.constants as c
import astropy.units as u
import numpy as np
from enterprise.pulsar import PintPulsar, Pulsar
from enterprise.signals.signal_base import PTA
from enterprise.signals.parameter import Uniform
from enterprise_extensions import models

from model import model_gwecc_1psr


def read_psr(datadir: str, parfile: str, timfile: str) -> PintPulsar:
    prefix = "".join(timfile.split(".")[:-1])
    psrfile = f"{datadir}/{prefix}.pkl"
    if not os.path.isfile(psrfile):
        psr = Pulsar(
            f"{datadir}/{parfile}", f"{datadir}/{timfile}", timing_package="pint"
        )
        with open(psrfile, "wb") as pkl:
            pickle.dump(psr, pkl)
    else:
        print(f"Loading pulsar from pickle {psrfile}")
        with open(psrfile, "rb") as pkl:
            psr = pickle.load(pkl)
            assert isinstance(psr, PintPulsar)

    return psr


# def decode_rednoise_params(rnamp: float, rnidx: float):
#     # https://github.com/nanograv/pint_pal/blob/main/src/pint_pal/noise_utils.py#L245
#     plredamp = np.log10(
#         rnamp / ((86400.0 * 365.24 * 1e6) / (2.0 * np.pi * np.sqrt(3.0)))
#     )
#     plredgam = -rnidx

#     return plredamp, plredgam


# def prepare_noise_dict(model: TimingModel) -> dict:
#     jname = model.PSR.value

#     noise_dict = {}

#     if "ScaleToaError" in model.components:
#         wn = model.components["ScaleToaError"]
#         for lbl, (tag, (tag_val,)) in wn.EFACs.items():
#             param_name = f"{jname}_{tag_val}_efac"
#             param_val = getattr(model, lbl).value
#             noise_dict[param_name] = param_val

#         for lbl, (tag, (tag_val,)) in wn.EQUADs.items():
#             # https://github.com/nanograv/pint_pal/blob/87c324c5248aaa1e0bf916dac21429c145b6fa1a/src/pint_pal/noise_utils.py#L334
#             param_name = f"{jname}_{tag_val}_log10_t2equad"
#             param_val = np.log10(1e-6 * getattr(model, lbl).value)
#             noise_dict[param_name] = param_val

#     if "EcorrNoise" in model.components:
#         ec = model.components["EcorrNoise"]
#         for lbl, (tag, (tag_val,)) in ec.ECORRs.items():
#             # https://github.com/nanograv/pint_pal/blob/87c324c5248aaa1e0bf916dac21429c145b6fa1a/src/pint_pal/noise_utils.py#L360
#             param_name = f"{jname}_{tag_val}_log10_ecorr"
#             param_val = np.log10(1e-6 * getattr(model, lbl).value)
#             noise_dict[param_name] = param_val

#     if hasattr(model, "RNAMP") and hasattr(model, "RNIDX"):
#         red_noise_log10_A, red_noise_gamma = decode_rednoise_params(
#             model.RNAMP.value, model.RNIDX.value
#         )
#         noise_dict[f"{jname}_red_noise_log10_A"] = red_noise_log10_A
#         noise_dict[f"{jname}_red_noise_gamma"] = red_noise_gamma

#     return noise_dict


def verify_noise_dict(psr: PintPulsar, noise_dict: dict):
    model = models.model_singlepsr_noise(
        psr,
        white_vary=True,
        red_var=True,
        noisedict=noise_dict,
        psr_model=True,
        tm_marg=True,
    )
    pta = PTA([model(psr)])
    assert set(pta.param_names) == set(noise_dict.keys())


def read_data(data_dir: str, par_file: str, tim_file: str, noise_dict_file: str):
    # model, toas = get_model_and_toas(parfile, timfile, planets=True, usepickle=True)
    psr = read_psr(data_dir, par_file, tim_file)

    with open(f"{data_dir}/{noise_dict_file}", "r") as ndf:
        noise_dict = json.load(ndf)
    verify_noise_dict(psr, noise_dict)

    return psr, noise_dict


def get_ecw_params(psr, settings):
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

    return ecw_params


def get_pta(
    psr,
    noise_dict,
    ecw_param_dict,
    noise_only=False,
    wn_vary=False,
    wn_normal_efac=False,
    rn_vary=True,
    rn_components=30,
) -> PTA:
    verify_noise_dict(psr, noise_dict)

    model = model_gwecc_1psr(
        noise_only=noise_only,
        wn_vary=wn_vary,
        wn_normal_efac=wn_normal_efac,
        rn_vary=rn_vary,
        rn_components=rn_components,
        ecw_param_dict=ecw_param_dict,
    )

    # model = models.model_singlepsr_noise(
    #     psr,
    #     white_vary=False,
    #     red_var=vary_red_noise,
    #     noisedict=noise_dict,
    #     psr_model=True,
    #     tm_marg=True
    # )

    # if not noise_only:
    #     wf = gwecc_1psr_block(**ecw_param_dict)
    #     model += wf

    print(f"pdist = {psr.pdist}")

    pta = PTA([model(psr)])
    pta.set_default_params(noise_dict)

    return pta


def prior_transform_fn(pta):
    mins = np.array([param.prior.func_kwargs["pmin"] for param in pta.params])
    maxs = np.array([param.prior.func_kwargs["pmax"] for param in pta.params])
    spans = maxs - mins

    def prior_transform(cube):
        return spans * cube + mins

    return prior_transform


def get_deltap_max(psr):
    Dp = psr.pdist[0]
    sigma_Dp = psr.pdist[1]
    Dp_max = Dp + sigma_Dp
    return (Dp_max * u.Unit("kpc") / c.c).to("year").value

def get_pta_from_settings(settings_file):
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

    ecw_params = get_ecw_params(psr, settings)

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

    return pta