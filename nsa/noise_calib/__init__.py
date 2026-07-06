"""IMX662-oriented noise calibration and forward synthesis (5-phase workflow).

Phases
------
1. **capture**   — organise bias / dark / flat-field calibration frames
2. **extract**   — isolate read, row, and shot noise samples
3. **fit**       — estimate {a, read_dist, row_dist, adc_bits}
4. **validate**  — held-out frame statistics vs simulation
5. **synthesize**— inject calibrated noise on clean ground-truth images
"""

from .model import NoiseModel, load_model, save_model
from .pipeline import run_calibration_pipeline
from .synthesize import synthesize_noisy, synthesize_pair

__all__ = [
    "NoiseModel",
    "load_model",
    "save_model",
    "run_calibration_pipeline",
    "synthesize_noisy",
    "synthesize_pair",
]
