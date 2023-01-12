from analysis import read_data, get_pta
from enterprise.signals.parameter import Uniform

import numpy as np

def main():
    prefix = "J1909-3744_NANOGrav_12yv4"
    parfile = f"{prefix}.gls.par"
    timfile = f"{prefix}.tim"
    psr, noise_dict = read_data(parfile, timfile, prefix)

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
        "gammap": Uniform(0, np.pi),
        "l0": Uniform(0, 2 * np.pi)(f"{name}_l0"),
        "lp": Uniform(0, 2 * np.pi),
        "tref": tref,
        "log10_zc": Uniform(-4, -3)(f"{name}_log10_zc"),
    }

    pta = get_pta(psr, noise_dict, prior_dict=priors)
    print(pta.param_names)


if __name__ == "__main__":
    main()