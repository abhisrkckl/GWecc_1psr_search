from analysis import read_data, get_pta, get_deltap_max
from enterprise.signals.parameter import Uniform

import numpy as np


def main():
    prefix = "J1909-3744_NANOGrav_12yv4"
    parfile = f"{prefix}.gls.par"
    timfile = f"{prefix}.tim"
    psr, noise_dict = read_data(parfile, timfile, prefix)

    name = "gwecc"

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
        "psrTerm": True,
        "spline": True,
    }

    pta = get_pta(psr, noise_dict, prior_dict=priors)
    print("Free parameters :", pta.param_names)

    # Make sure that the PTA object works.
    # This also triggers JIT compilation of julia code and the caching of ENTERPRISE computations.
    x0 = np.array([p.sample() for p in pta.params])
    print("Log-likelihood at", x0, "is", pta.get_lnlikelihood(x0))


if __name__ == "__main__":
    main()
