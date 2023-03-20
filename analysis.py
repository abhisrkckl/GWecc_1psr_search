from pint.models import get_model_and_toas, TimingModel
from pint.toa import TOAs
from enterprise.pulsar import Pulsar, PintPulsar
from enterprise.signals.signal_base import PTA
from enterprise_extensions import models
from enterprise_gwecc import gwecc_1psr_block

import json
import os
import pickle
import numpy as np
import astropy.units as u
import astropy.constants as c


def read_psr(datadir: str, parfile: str, timfile: str) -> PintPulsar:
    prefix = "".join(timfile.split(".")[:-1])
    psrfile = f"{datadir}/{prefix}.pkl"
    if not os.path.isfile(psrfile):
        psr = Pulsar(f"{datadir}/{parfile}", f"{datadir}/{timfile}", timing_package="pint")
        with open(psrfile, "wb") as pkl:
            pickle.dump(psr, pkl)
    else:
        print(f"Loading pulsar from pickle {psrfile}")
        with open(psrfile, "rb") as pkl:
            psr = pickle.load(pkl)
            assert isinstance(psr, PintPulsar)

    return psr


def decode_rednoise_params(rnamp: float, rnidx: float):
    # https://github.com/nanograv/pint_pal/blob/main/src/pint_pal/noise_utils.py#L245
    plredamp = np.log10(
        rnamp / ((86400.0 * 365.24 * 1e6) / (2.0 * np.pi * np.sqrt(3.0)))
    )
    plredgam = -rnidx

    return plredamp, plredgam


def prepare_noise_dict(model: TimingModel) -> dict:
    jname = model.PSR.value

    noise_dict = {}

    if "ScaleToaError" in model.components:
        wn = model.components["ScaleToaError"]
        for lbl, (tag, (tag_val,)) in wn.EFACs.items():
            param_name = f"{jname}_{tag_val}_efac"
            param_val = getattr(model, lbl).value
            noise_dict[param_name] = param_val

        for lbl, (tag, (tag_val,)) in wn.EQUADs.items():
            # https://github.com/nanograv/pint_pal/blob/87c324c5248aaa1e0bf916dac21429c145b6fa1a/src/pint_pal/noise_utils.py#L334
            param_name = f"{jname}_{tag_val}_log10_t2equad"
            param_val = np.log10(1e-6 * getattr(model, lbl).value)
            noise_dict[param_name] = param_val

    if "EcorrNoise" in model.components:
        ec = model.components["EcorrNoise"]
        for lbl, (tag, (tag_val,)) in ec.ECORRs.items():
            # https://github.com/nanograv/pint_pal/blob/87c324c5248aaa1e0bf916dac21429c145b6fa1a/src/pint_pal/noise_utils.py#L360
            param_name = f"{jname}_{tag_val}_log10_ecorr"
            param_val = np.log10(1e-6 * getattr(model, lbl).value)
            noise_dict[param_name] = param_val

    if hasattr(model, "RNAMP") and hasattr(model, "RNIDX"):
        red_noise_log10_A, red_noise_gamma = decode_rednoise_params(
            model.RNAMP.value, model.RNIDX.value
        )
        noise_dict[f"{jname}_red_noise_log10_A"] = red_noise_log10_A
        noise_dict[f"{jname}_red_noise_gamma"] = red_noise_gamma

    return noise_dict


def verify_noise_dict(psr: PintPulsar, noise_dict: dict):
    model = models.model_singlepsr_noise(
        psr, white_vary=True, red_var=True, noisedict=noise_dict, psr_model=True
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


def get_pta(psr, vary_red_noise, noise_dict, ecw_param_dict) -> PTA:
    verify_noise_dict(psr, noise_dict)

    model = models.model_singlepsr_noise(
        psr, white_vary=False, red_var=vary_red_noise, noisedict=noise_dict, psr_model=True
    )

    wf = gwecc_1psr_block(**ecw_param_dict)
    model += wf

    print(f"pdist = {psr.pdist}")

    pta = PTA([model(psr)])
    pta.set_default_params(noise_dict)

    return pta

def prior_transform_fn(pta):
    mins = np.array([param.prior.func_kwargs['pmin'] for param in pta.params])
    maxs = np.array([param.prior.func_kwargs['pmax'] for param in pta.params])
    spans = maxs-mins
    
    def prior_transform(cube):
        return spans*cube + mins
    
    return prior_transform

def get_deltap_max(psr):
    Dp = psr.pdist[0]
    sigma_Dp = psr.pdist[1]
    Dp_max = Dp + sigma_Dp
    return (Dp_max * u.Unit("kpc") / c.c).to("year").value