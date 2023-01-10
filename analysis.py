from pint.models import get_model_and_toas, TimingModel
from pint.toa import TOAs
from enterprise.pulsar import Pulsar, PintPulsar
from enterprise.signals.signal_base import PTA, function as enterprise_function
from enterprise.signals.parameter import Uniform
from enterprise.signals.deterministic_signals import Deterministic
from enterprise_extensions import models

import os
import pickle
import numpy as np

from juliacall import Main as jl

jl.seval("using GWecc")


@enterprise_function
def eccentric_pta_signal_planck18_1psr(
    toas,
    pdist,
    alpha,
    psi,
    cos_inc,
    log10_M,
    eta,
    log10_F,
    e0,
    gamma0,
    gammap,
    l0,
    lp,
    tref,
    log10_zc,
    psrTerm=False,
):
    return jl.eccentric_pta_signal_planck18_1psr(
        toas,
        pdist,
        alpha,
        psi,
        cos_inc,
        log10_M,
        eta,
        log10_F,
        e0,
        gamma0,
        gammap,
        l0,
        lp,
        tref,
        log10_zc,
        psrTerm,
    )


def read_psr(model: TimingModel, toas: TOAs, prefix: str):
    psrfile = f"{prefix}.pkl"
    if not os.path.isfile(psrfile):
        psr = Pulsar(model, toas)
        with open(psrfile, "wb") as pkl:
            pickle.dump(psr, pkl)
    else:
        with open(psrfile, "rb") as pkl:
            psr = pickle.load(pkl)
            assert isinstance(psr, PintPulsar)

    return psr


def decode_rednoise_params(rnamp, rnidx):
    # https://github.com/nanograv/pint_pal/blob/main/src/pint_pal/noise_utils.py#L245
    plredamp = np.log10(
        rnamp / ((86400.0 * 365.24 * 1e6) / (2.0 * np.pi * np.sqrt(3.0)))
    )
    plredgam = -rnidx

    return plredamp, plredgam


def prepare_noise_dict(model: TimingModel):
    jname = model.PSR.value

    noise_dict = {}

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

    ec = model.components["EcorrNoise"]
    for lbl, (tag, (tag_val,)) in ec.ECORRs.items():
        # https://github.com/nanograv/pint_pal/blob/87c324c5248aaa1e0bf916dac21429c145b6fa1a/src/pint_pal/noise_utils.py#L360
        param_name = f"{jname}_{tag_val}_log10_ecorr"
        param_val = np.log10(1e-6 * getattr(model, lbl).value)
        noise_dict[param_name] = param_val

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


def read_data(prefix: str):
    parfile = f"{prefix}.gls.par"
    timfile = f"{prefix}.tim"

    model, toas = get_model_and_toas(parfile, timfile, planets=True, usepickle=True)
    psr = read_psr(model, toas, prefix)
    noise_dict = prepare_noise_dict(model)

    verify_noise_dict(psr, noise_dict)

    return psr, noise_dict


def get_gwecc_1psr_priors(tref, name="gwecc"):
    return {
        "alpha": Uniform(0, 1)(f"{name}_alpha"),
        "psi": Uniform(0, np.pi)(f"{name}_psi"),
        "cos_inc": Uniform(-1, 1)(f"{name}_cos_inc"),
        "log10_M": Uniform(6, 9)(f"{name}_log10_M"),
        "eta": Uniform(0, 0.25)(f"{name}_eta"),
        "log10_F": Uniform(-9, -7)(f"{name}_log10_F"),
        "e0": Uniform(0.01, 0.8)(f"{name}_e0"),
        "gamma0": Uniform(0, np.pi)(f"{name}_gamma0"),
        "gammap": Uniform(0, np.pi)(f"{name}_gammap"),
        "l0": Uniform(0, 2 * np.pi)(f"{name}_l0"),
        "lp": Uniform(0, 2 * np.pi)(f"{name}_lp"),
        "tref": tref,
        "log10_zc": Uniform(-4, -3)(f"{name}_log10_zc"),
    }


def test_gwecc_func():
    alpha = 0.3
    psi = 1.1
    cos_inc = 0.4
    log10_M = 7.0
    eta = 0.2
    log10_F = -8.0
    e0 = 0.3
    gamma0 = np.pi / 2
    gammap = np.pi / 2
    l0 = np.pi / 3
    lp = np.pi / 3
    log10_zc = -3.5

    year = 365.25 * 24 * 3600
    toas = np.linspace(0, 5 * year, 100)
    tref = max(toas)

    pdist = 4.0

    for psrTerm in [True, False]:
        ss = eccentric_pta_signal_planck18_1psr(
            toas,
            pdist,
            alpha,
            psi,
            cos_inc,
            log10_M,
            eta,
            log10_F,
            e0,
            gamma0,
            gammap,
            l0,
            lp,
            tref,
            log10_zc,
            psrTerm=False,
        )
        assert np.all(np.isfinite(ss))


def get_pta(
    psr, noise_dict, signal_func=eccentric_pta_signal_planck18_1psr, prior_dict=None
) -> PTA:
    verify_noise_dict(psr, noise_dict)
    test_gwecc_func()

    model = models.model_singlepsr_noise(
        psr, white_vary=False, red_var=False, noisedict=noise_dict, psr_model=True
    )

    signal_priors = (
        get_gwecc_1psr_priors(tref=max(psr.toas)) if prior_dict is None else prior_dict
    )

    wf = Deterministic(signal_func(**signal_priors), name="gwecc")
    model += wf

    print(f"pdist = {psr.pdist}")

    pta = PTA([model(psr)])
    pta.set_default_params(noise_dict)

    return pta


def main():
    prefix = "J1909-3744_NANOGrav_12yv4"
    psr, noise_dict = read_data(prefix)

    name = "gwecc"

    tref = max(psr.toas)
    priors = {
        "alpha": Uniform(0, 1)(f"{name}_alpha"),
        "psi": Uniform(0, np.pi)(f"{name}_psi"),
        "cos_inc": Uniform(-1, 1)(f"{name}_cos_inc"),
        "log10_M": Uniform(6, 9)(f"{name}_log10_M"),
        "eta": Uniform(0, 0.25)(f"{name}_eta"),
        "log10_F": Uniform(-9, -7)(f"{name}_log10_F"),
        "e0": Uniform(0.01, 0.8)(f"{name}_e0"),
        "gamma0": Uniform(0, np.pi)(f"{name}_gamma0"),
        "gammap": Uniform(0, np.pi)(f"{name}_gammap"),
        "l0": Uniform(0, 2 * np.pi)(f"{name}_l0"),
        "lp": Uniform(0, 2 * np.pi)(f"{name}_lp"),
        "tref": tref,
        "log10_zc": Uniform(-4, -3)(f"{name}_log10_zc"),
    }

    pta = get_pta(psr, noise_dict, prior_dict=priors)
    print(pta.param_names)


if __name__ == "__main__":
    main()
