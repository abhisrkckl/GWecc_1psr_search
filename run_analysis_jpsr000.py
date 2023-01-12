from analysis import read_data
from enterprise.signals.parameter import Uniform
from enterprise.signals.gp_signals import MarginalizingTimingModel
from enterprise.signals.signal_base import PTA
from enterprise_gwecc import gwecc_1psr_block

import numpy as np


def main():
    prefix = "JPSR00_simulate"
    parfile = f"{prefix}.par"
    timfile = f"{prefix}.tim"
    psr = read_data(parfile, timfile, prefix, noise_dict=False)

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

    tm = MarginalizingTimingModel()
    wf = gwecc_1psr_block(**priors)
    model = tm + wf

    pta = PTA([model(psr)])

    print(pta.param_names)


if __name__ == "__main__":
    main()
