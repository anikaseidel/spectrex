"""specTrex — Grism spectra extraction in crowded regions.

Public API
----------
InstrumentConfig
    Instrument configuration loader (grism, filter, sensitivity curves).
EigenspectraBasis
    PCA eigenspectra basis for spectral representation.
ForwardOperatorProtocol
    Protocol that any forward operator must satisfy.
SciPySparseOperator
    scipy-sparse forward operator (Phase 1).
NoiseModel
    Poisson + read-noise model.
SpectralSolver
    LSQR/LSMR solver for grism deconvolution.
"""

from spectrex._version import version as __version__
from spectrex.basis import EigenspectraBasis
from spectrex.instrument import InstrumentConfig
from spectrex.operator import ForwardOperatorProtocol, SciPySparseOperator
from spectrex.solver import NoiseModel, SpectralSolver

__all__ = [
    "__version__",
    "InstrumentConfig",
    "EigenspectraBasis",
    "ForwardOperatorProtocol",
    "SciPySparseOperator",
    "NoiseModel",
    "SpectralSolver",
]
