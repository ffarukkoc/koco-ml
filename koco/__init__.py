"""koco - toolkit for building regression and binary classification models."""

from . import analysis, feature_selection, general, guides, metrics, modelling, preprocessing, registry, scoring
from .analysis import *
from .feature_selection import *
from .general import *
from .guides import *
from .metrics import *
from .modelling import *
from .preprocessing import *
from .scoring import *

__version__ = "0.1.0"

# Each module lists its public names in its own __all__; adding a function only needs that one line.
__all__ = []
__all__ += general.__all__
__all__ += analysis.__all__
__all__ += preprocessing.__all__
__all__ += feature_selection.__all__
__all__ += metrics.__all__
__all__ += modelling.__all__
__all__ += scoring.__all__
__all__ += guides.__all__

_Duplicates = sorted({Name for Name in __all__ if __all__.count(Name) > 1})
if _Duplicates:
    raise ImportError(f"Public names exported by more than one koco module: {_Duplicates}")
del _Duplicates
