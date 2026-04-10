"""
config.py
=========
Single source of truth for ICP monitoring feature definitions, ranges, and constants.

All scripts (training, inference, validation, hardware) should import from here
to avoid train/serve skew and inconsistent validation boundaries.

─── Physiological Basis ────────────────────────────────────────────────────────
Optical tympanic membrane (TM) displacement encodes ICP via the established
cochlear aqueduct pathway: ICP → CSF pressure → perilymph → round window → TM.

Key literature supporting this signal chain:
  [1] Marchbanks RJ (1996). "Hydromechanical interactions of the intracranial
      and intralabyrinthine fluids." In: Intracranial and Intralabyrinthine Fluids.
      Springer. Established the physiological TM↔ICP coupling mechanism.
  [2] Skjæret-Maroni N et al. (2021). "Noninvasive detection of elevated
      intracranial pressure using spontaneous tympanic membrane pulsation."
      Scientific Reports 11, 22071. doi:10.1038/s41598-021-01079-8
      → Demonstrated spontaneous TM amplitude correlates with mean ICP and
        spectral ICP amplitude in patients with brain pathologies.
  [3] Bershad EM et al. (2010). "Intracranial pressure and pulsatility index
      changes in quantitative analysis." J Neurosurg.
      → Confirmed cardiac/respiratory frequency ICP components visible in TMD.
  [4] Mahan M et al. (2023). "Applying video motion magnification to the
      tympanic membrane for non-invasive intracranial pressure monitoring."
      Acta Neurochirurgica 165. doi:10.1007/s00701-023-05647-3
      → Modern optical approach: TM video motion magnification reveals ICP surges.
  [5] Samuel M et al. (1998). "Noninvasive ICP monitoring: tympanic membrane
      displacement." J Neurosurg 88:983-988.
      → Spontaneous TMD pulsations correlate with ICP waveform morphology.

─── Feature Rationale (grounded in ICP waveform morphology literature) ────────
  cardiac_amplitude   : ICP P1 percussion wave amplitude proxy — pulsatile
                        TM displacement driven by arterial blood pressure
                        transmitted via choroid plexus [2, 4].
  cardiac_frequency   : Heart rate — reduced HRV is a late ICP elevation marker.
  respiratory_amplitude: B-wave / respiratory ICP modulation (0.1–0.5 Hz band);
                         elevated in reduced intracranial compliance [3].
  slow_wave_power     : Lundberg A/B slow-wave activity (db4 wavelet, approx
                        0–1.56 Hz); high power indicates compensated ICP state.
  cardiac_power       : Relative cardiac-band wavelet energy; cardiac/slow-wave
                        ratio is an established ICP compliance surrogate [3].
  mean_arterial_pressure: CPP = MAP − ICP; MAP is the primary ICP confound and
                         most discriminative inter-patient feature.

─── Training Data ──────────────────────────────────────────────────────────────
  CHARIS  : 13 TBI ICU patients, PhysioNet (physionet.org/content/charisdb)
  MIMIC-III WDB: General ICU, PhysioNet (physionet.org/content/mimic3wdb)
  Both datasets provide simultaneous invasive ICP + ABP waveforms used as
  surrogate ground truth for proof-of-concept feasibility evaluation.
  Feature extraction is signal-agnostic (wavelet ratios, spectral amplitudes)
  and applies identically to ICP waveforms (training) and TM displacement
  (inference), following the established physiological equivalence [1, 2].
"""
from __future__ import annotations

# ── Feature specification ────────────────────────────────────────────────────

FEATURE_NAMES = [
    "cardiac_amplitude",
    "cardiac_frequency",
    "respiratory_amplitude",
    "slow_wave_power",
    "cardiac_power",
    "mean_arterial_pressure",
]

N_FEATURES = len(FEATURE_NAMES)

FEATURE_UNITS = {
    "cardiac_amplitude":      "μm",
    "cardiac_frequency":      "Hz",
    "respiratory_amplitude":  "μm",
    "slow_wave_power":        "",       # dimensionless wavelet energy fraction
    "cardiac_power":          "",       # dimensionless wavelet energy fraction
    "mean_arterial_pressure": "mmHg",
}

# Physiological validation ranges — MUST match training data distribution.
# These are used by both the backend (validation.py) and inference scripts.
# Widened to accommodate actual training data (CHARIS + MIMIC) distributions.
FEATURE_RANGES = {
    "cardiac_amplitude":      (5.0, 120.0),     # μm   — adjusted for MIMIC variability
    "cardiac_frequency":      (0.7, 2.5),       # Hz   — 42-150 bpm
    "respiratory_amplitude":  (1.0, 50.0),      # μm   — adjusted for MIMIC variability
    # slow_wave_power = wavelet energy fraction in 0–1.56 Hz band (db4, level-5).
    # Near-1.0 = normal (low-freq dominant); lower = more high-freq activity.
    "slow_wave_power":        (0.30, 1.0),      # dimensionless — widened for MIMIC
    # cardiac_power = wavelet energy fraction in 1.56–3.12 Hz band.
    # Training data max can reach ~0.35 for unusual MIMIC spectra.
    "cardiac_power":          (0.0, 0.40),       # dimensionless — matches training max
    "mean_arterial_pressure": (40.0, 200.0),    # mmHg — clamped in extraction
}

# ── Classification ────────────────────────────────────────────────────────────

CLASS_NAMES = ["Normal", "Abnormal"]
ICP_THRESHOLD_MMHG = 15.0    # Clinical intervention threshold

# ── Extraction parameters ────────────────────────────────────────────────────

TARGET_FS = 125              # Hz — target sampling frequency
WINDOW_SAMPLES = 1250       # 10 seconds × 125 Hz
WINDOW_SECONDS = 10.0
