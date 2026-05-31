# Non-Invasive Intracranial Pressure Monitoring System

<div align="center">

![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen)
![LOPO AUC](https://img.shields.io/badge/LOPO%20AUC-0.9611-blue)
![MIMIC Accuracy](https://img.shields.io/badge/MIMIC--III%20Accuracy-89.8%25-blue)
![Patients](https://img.shields.io/badge/Hardware%20Subjects-16-orange)
![Papers](https://img.shields.io/badge/Papers-3%20Planned-purple)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**A non-invasive, optical tympanic membrane sensor system for real-time ICP anomaly detection and 30-minute advance elevation forecasting.**

*Capstone Research Project — Thapar Institute of Engineering and Technology*

</div>

---

## The Clinical Problem

> **Every year, over 69 million people sustain traumatic brain injury (TBI) worldwide.** Elevated intracranial pressure (ICP > 20 mmHg) is the leading cause of secondary brain injury and death. The current gold standard — drilling a hole in the skull and inserting a pressure bolt — is invasive, risky, and available only in ICU settings.

**We built a non-invasive alternative: an optical sensor placed in the ear canal.**

The system does two things no existing non-invasive approach does together:
1. **Real-time detection** — Is ICP elevated *right now*?
2. **30-minute forecasting** — Will ICP *become* critical in the next 30 minutes?

---

## The Science: How the Ear Reflects Brain Pressure

The tympanic membrane (TM) is hydraulically coupled to intracranial pressure through a well-established anatomical pathway:

```
ICP Change
    │
    ▼
CSF pressure change in cochlear aqueduct
    │
    ▼
Perilymph pressure change in scala tympani
    │
    ▼
Round window membrane displacement
    │
    ▼
Tympanic membrane micro-displacement  ← Our sensor detects this
```

> **Key insight:** When ICP rises, the TM stiffens and its acoustic/optical properties change. These changes are measurable — they carry the same frequency signatures (cardiac pulsations, respiratory modulation, slow waves) as the ICP waveform itself.

**Published basis:**
- Ragauskas et al. (2005) — TM displacement correlates with ICP
- Gwisdalla et al. (2012) — Ear canal pressure reflects ICP dynamics
- Aaslid et al. (1989) — Non-invasive cerebrovascular compliance measurement

---

## Three-Paper Research Strategy

This project is structured as three interconnected publications, each with a distinct contribution:

```mermaid
graph LR
    P1["📡 Paper 1\nHardware Sensor\nIEEE Sensors Journal"] --> P2
    P2["🤖 Paper 2\nXGBoost Detection\nSensors / BSPC"] --> P3
    P3["🧠 Paper 3\nBiLSTM Forecasting\nIEEE JBHI"]

    style P1 fill:#2d6a4f,color:#fff
    style P2 fill:#1d3557,color:#fff
    style P3 fill:#6d2b9f,color:#fff
```

| | Paper 1 | Paper 2 | Paper 3 |
|---|---|---|---|
| **Contribution** | Novel TM optical sensor | Real-time ICP anomaly detection | 30-min advance ICP forecasting |
| **Model** | None (sensor validation) | XGBoost + QuantileTransformer | BiLSTM + Self-Attention |
| **Target** | IEEE Sensors Journal | Sensors (MDPI) / BSPC | IEEE JBHI / npj Digital Medicine |
| **Key Result** | Age gradient 0.6%→72.2% | LOPO AUC 0.9611 | 30-min advance warning |

---

## System Architecture

```mermaid
graph TD
    A["👂 Ear Canal\nOptical TM Sensor"] --> B["📡 Hardware\nIR + Displacement\n50 Hz, 4 sessions"]
    B --> C["⚙️ Feature Extraction\n5 ICP-correlated features\n10-sec windows, 5-sec stride"]

    C --> D{"Which Task?"}

    D --> E["🎯 XGBoost\nCurrent State\nDetection"]
    D --> F["🧠 BiLSTM\n30-min\nForecasting"]

    E --> G["ICP Normal / Elevated\nReal-time alert"]
    F --> H["ICP will rise\nin 30 minutes\nAdvance warning"]

    style A fill:#e76f51,color:#fff
    style B fill:#e9c46a,color:#333
    style C fill:#2a9d8f,color:#fff
    style E fill:#1d3557,color:#fff
    style F fill:#6d2b9f,color:#fff
    style G fill:#2d6a4f,color:#fff
    style H fill:#9d0208,color:#fff
```

---

## Hardware Protocol

Each subject undergoes a standardised 4-session recording protocol:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Recording Protocol                           │
├──────────────┬──────────┬───────────────────────────────────────┤
│ Session      │ Duration │ Purpose                               │
├──────────────┼──────────┼───────────────────────────────────────┤
│ 0  Supine    │  10 min  │ Baseline resting state                │
│ 1  Head +30° │   5 min  │ ICP reduction (postural)              │
│ 2  Head -10° │   5 min  │ ICP elevation (postural)              │
│ 3  Valsalva  │  ~7 min  │ Controlled transient ICP spike        │
├──────────────┴──────────┴───────────────────────────────────────┤
│ Total: ~27 minutes per subject  |  81,000 samples @ 50 Hz      │
└─────────────────────────────────────────────────────────────────┘
```

**16 subjects collected — ages 8 to 75:**

| Subject Group | N | Age Range | Profile |
|---|---|---|---|
| Children | 2 | 8–12 | Healthy |
| Teenagers | 2 | 16–21 | Healthy |
| Adults | 6 | 40–55 | Healthy |
| Elderly (healthy) | 3 | 65–75 | No comorbidities |
| Elderly (comorbid) | 2 | 72–75 | Hypertension / Diabetes |
| Pathological | 1 | 65–75 | Prior haemorrhage |

---

## Feature Extraction Pipeline

The same 5 features are extracted from every 10-second window (500 samples @ 50 Hz):

```mermaid
graph TD
    Raw["Raw Signal Window\n500 samples @ 50 Hz\n10 seconds"] --> Detrend["Detrend\nRemove DC offset"]

    Detrend --> Card["Bandpass Filter\n1.0–2.5 Hz\nCardiac band"]
    Detrend --> Resp["Bandpass Filter\n0.1–0.5 Hz\nRespiratory band"]
    Detrend --> Wave["Wavelet\nDecompose db4\nlevel 5"]
    Detrend --> FFT["FFT\n0.7–2.5 Hz\nDominant freq"]

    Card --> F1["cardiac_amplitude\nP99–P1 of filtered signal"]
    FFT  --> F2["cardiac_frequency\nDominant cardiac Hz"]
    Resp --> F3["respiratory_amplitude\nP99–P1 of filtered signal"]
    Wave --> F4["slow_wave_power\nEnergy ratio: cA5 / total"]
    Wave --> F5["cardiac_power\nEnergy ratio: cD4 / total"]

    style Raw fill:#e76f51,color:#fff
    style F1 fill:#2d6a4f,color:#fff
    style F2 fill:#2d6a4f,color:#fff
    style F3 fill:#2d6a4f,color:#fff
    style F4 fill:#2d6a4f,color:#fff
    style F5 fill:#2d6a4f,color:#fff
```

**Why these 5 features?**

| Feature | Physiology | ICP Link |
|---|---|---|
| `cardiac_amplitude` | Magnitude of cardiac ICP pulsations | Higher ICP → higher pulse pressure amplitude |
| `cardiac_frequency` | Heart rate extracted from ICP signal | Dysrhythmia correlates with intracranial hypertension |
| `respiratory_amplitude` | Breathing-induced ICP oscillations | Elevated ICP alters respiratory modulation |
| `slow_wave_power` | Lundberg slow waves (0–0.5 Hz) | Pathological slow waves appear with elevated ICP |
| `cardiac_power` | Fraction of signal energy in cardiac band | Shifts with cerebrovascular compliance changes |

---

## Paper 2: XGBoost Real-Time Detection

### Training Pipeline

```mermaid
graph TD
    C["CHARIS Database\n13 TBI patients\n915,137 windows\nOpen access — PhysioNet"] --> FE["Feature Extraction\n5 features per 10-sec window"]
    FE --> Split["Patient-Level Split\n70% Train / 10% Val / 20% Test\nNo window-level leakage"]
    Split --> QT["QuantileTransformer\nFitted on TRAIN ONLY\nMaps features → N(0,1)"]
    QT --> SMOTE["SMOTE\nWithin-patient oversampling\nBalances normal / abnormal"]
    SMOTE --> XGB["XGBoost\nGPU-accelerated CUDA\nEarly stopping on val AUC"]
    XGB --> LOPO["LOPO Cross-Validation\n13 folds, per-fold QT fit\nLeakage-free patient generalisation"]
    LOPO --> MIMIC["MIMIC-III External Validation\n12 independent ICU patients\nReal invasive ICP reference"]

    style C fill:#457b9d,color:#fff
    style LOPO fill:#e9c46a,color:#333
    style MIMIC fill:#2d6a4f,color:#fff
```

### Results

```
┌─────────────────────────────────────────────────────────────────┐
│              XGBoost Classification Results                     │
├─────────────────────────────────────────────┬───────────────────┤
│ CHARIS Test AUC                             │      0.9792       │
│ CHARIS Train-Test Gap                       │      0.0106       │
│ LOPO AUC  (gold standard)                   │  0.9611 ± 0.058   │
│ LOPO 95% CI                                 │ [0.9242, 0.9852]  │
│ F1 Score                                    │      0.8040       │
│ Sensitivity (catches true abnormals)        │      87.6%        │
│ Specificity (clears true normals)           │      95.6%        │
├─────────────────────────────────────────────┴───────────────────┤
│              MIMIC-III Independent Validation                   │
│              (different hospital, real ICP bolts)               │
├─────────────────────────────────────────────┬───────────────────┤
│ Patients / Windows                          │   12 / 4,027      │
│ Pearson r  (P vs ICP mmHg)                  │     +0.7131       │
│ Spearman ρ (rank correlation)               │     +0.8060       │
│ Sensitivity                                 │      85.8%        │
│ Specificity                                 │      90.0%        │
│ Overall Accuracy                            │      89.8%        │
└─────────────────────────────────────────────┴───────────────────┘
```

### Validation Pyramid

```
                    ┌─────────────────────┐
                    │    MIMIC-III        │  ← Strongest: external, independent,
                    │  12 patients        │    real invasive ICP, different hospital
                    │  4,027 windows      │
                    │  89.8% accuracy     │
                  ┌─┴─────────────────────┴─┐
                  │     CHARIS LOPO CV      │  ← Gold standard: held-out patient
                  │     13 folds            │    never in training for that fold
                  │     AUC 0.9611          │
                ┌─┴─────────────────────────┴─┐
                │      CHARIS Test Split      │  ← Standard: 3 held-out patients
                │      3 patients             │
                │      AUC 0.9792             │
                └─────────────────────────────┘
```

### Hardware Age Gradient (Physiological Validation)

```
Age  8F  ████░░░░░░░░░░░░░░░░░░░░░░░░░░  0.6%   Normal child
Age 12M  ████████░░░░░░░░░░░░░░░░░░░░░░  2.2%   Normal child
Age 16M  █████████████░░░░░░░░░░░░░░░░░  4.4%   Normal teen
Age 19F  █████████████████████░░░░░░░░░  6.6%   Healthy young adult
Age 40F  ████████████░░░░░░░░░░░░░░░░░░  3.8%   Healthy adult
Age 40M  ████████████████████░░░░░░░░░░  7.5%   Healthy adult
Age 47F  ████████████████████░░░░░░░░░░  8.2%   Healthy adult
Age 50M  ████████████████████████░░░░░░ 12.6%   Mild elevation (age-related)
Age 55M  ████████████████████░░░░░░░░░░  8.5%   Healthy adult
Age 65+  ████████████████████████████░░ 27.0%   Elderly (normal)
Age 75F  ██████████████████████████████ 40.9%   Elderly (no comorbidity)
Age 72F  ████████████████████████████████ 46.2% Elderly + HTN/DM
Age 65+H ██████████████████████████████████ 65.6% Elderly + haemorrhage
Age 75M  ████████████████████████████████████ 72.2% Elderly + HTN/DM (highest)
```

> **Key finding:** Same age (75), different pathology → 31.3% gap. Model detects **cerebrovascular compliance**, not just age.

---

## Paper 3: BiLSTM 30-Minute ICP Forecasting

### The Forecasting Task

```mermaid
graph LR
    H["Past 5 minutes\n60 windows\n5 features each"] --> M["BiLSTM\n+ Self-Attention"]
    M --> P["P ICP will exceed\n20 mmHg in the\nnext 30 minutes"]

    T0["t = now"] --- T1["t + 30 min"]
    style H fill:#1d3557,color:#fff
    style M fill:#6d2b9f,color:#fff
    style P fill:#9d0208,color:#fff
```

**Why this matters clinically:** Current ICP monitors alarm only when ICP is *already* elevated — by then, irreversible damage may have begun. A 30-minute advance warning gives clinicians time to:
- Adjust head position
- Administer osmotic therapy (mannitol)
- Prepare for emergency intervention

### Label Generation

For each window `i` in patient `p`:

```
y_forecast[i] = 1  if  ANY window in [i+1 ... i+360] is abnormal
              = 0  otherwise
              = -1 (excluded) for the last 360 windows of each patient
```

> 360 windows × 5-second stride = **30 minutes of look-ahead**. This is computed per-patient with no cross-patient leakage.

### Architecture

```mermaid
graph TD
    In["Input\nbatch × 60 × 5\n5 min history, 5 features"] --> L1["BiLSTM Layer 1\nhidden=64, bidirectional\noutput: batch × 60 × 128"]
    L1 --> Drop1["Dropout 0.3"]
    Drop1 --> L2["BiLSTM Layer 2\nhidden=64, bidirectional\noutput: batch × 60 × 128"]
    L2 --> Attn["Self-Attention\nBahdanau-style\nlearns which timesteps matter"]
    Attn --> Ctx["Context Vector\nbatch × 128"]
    Ctx --> Norm["LayerNorm"]
    Norm --> Drop2["Dropout 0.3"]
    Drop2 --> FC1["Linear 128→32\nReLU"]
    FC1 --> Out["Linear 32→1\nSigmoid → P forecast"]

    style In fill:#1d3557,color:#fff
    style Attn fill:#e9c46a,color:#333
    style Out fill:#9d0208,color:#fff
```

**Why BiLSTM + Attention over simple LSTM?**
- **Bidirectional:** Captures both rising and falling trends within the 5-min window
- **Self-Attention:** Learns to weight critical moments (e.g., a spike 2 min ago matters more than baseline)
- **Forecasting ≠ Classification:** The model sees past patterns, not just the current state

### Training Configuration

```
Horizon  : 360 windows = 30 minutes
History  : 60 windows  = 5 minutes
Sequences: 132,930 training sequences
Batch    : 512
Epochs   : 50 with cosine annealing LR
Optimizer: Adam, lr=1e-3, weight_decay=1e-4
Loss     : BCEWithLogitsLoss with pos_weight (class-balanced)
```

---

## Data Sources

```mermaid
graph TD
    CH["CHARIS Database\nPhysioNet — Open Access\n13 TBI patients, 50 Hz\nECG + ABP + ICP waveforms\nUsed for: TRAINING"]
    HW["Hardware Dataset\nIn-house collection\n16 subjects, ages 8–75\nOptical TM sensor\nUsed for: VALIDATION"]
    MI["MIMIC-III Waveform DB\nPhysioNet — Credentialed\n12 ICU patients\nReal invasive ICP bolts\nUsed for: INDEPENDENT VALIDATION"]

    CH --> Model["Trained Model"]
    Model --> HW
    Model --> MI

    style CH fill:#457b9d,color:#fff
    style HW fill:#2d6a4f,color:#fff
    style MI fill:#e76f51,color:#fff
```

| Dataset | Source | N Patients | Signal | Role |
|---|---|---|---|---|
| CHARIS | PhysioNet (open) | 13 | ICP + ABP + ECG | Training |
| Hardware | In-house | 16 | TM optical sensor | Hardware validation |
| MIMIC-III | PhysioNet (credentialed) | 12 | Invasive ICP + ABP | Independent validation |

---

## Key Results Summary

| Metric | Value | Significance |
|---|---|---|
| LOPO AUC | **0.9611 ± 0.058** | Patient-level generalisation proven |
| MIMIC Accuracy | **89.8%** | Independent hospital, real ICP bolts |
| MIMIC Spearman ρ | **+0.806** | Model output tracks continuous ICP mmHg |
| Extreme windows | **20 / 20 correct** | ICP=0→P≈0, ICP=46→P≈1 |
| Hardware gradient | **0.6% → 72.2%** | Physiologically ordered across all 16 subjects |
| Comorbidity gap | **+31.3%** | Same age, different pathology — model detects it |

---

## Repo Structure

```
Pran/
│
├── full_pipeline_qt.py      # Paper 2: XGBoost detection (main pipeline)
├── bilstm_forecast.py       # Paper 3: BiLSTM 30-min forecasting
├── mimic_validate.py        # MIMIC-III independent validation
├── regen_cache.py           # Regenerate CHARIS feature cache
├── show_results.py          # Visualisation and results display
│
├── hw-tests/                # Hardware CSV recordings (16 subjects)
│   ├── icp_1_27min.csv
│   ├── icp_2.csv
│   └── icp_{N}_{age}_{sex}.csv   # Auto-detected by pipeline
│
├── data/
│   └── raw/charis/          # CHARIS WFDB files (charis1–charis13)
│
├── models/
│   ├── xgb_qt.json          # Trained XGBoost model
│   ├── qt_scaler.pkl        # QuantileTransformer (XGBoost)
│   └── bilstm_forecaster.pt # Trained BiLSTM forecaster (after training)
│
└── results/
    ├── audit/cache/         # Precomputed CHARIS features (X, y, pid)
    ├── qt_pipeline/         # XGBoost results and plots
    ├── bilstm_forecast/     # BiLSTM forecasting results
    └── mimic_validation/    # MIMIC-III validation results
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/your-repo/Pran.git
cd Pran
pip install xgboost scikit-learn imbalanced-learn torch pywt wfdb scipy matplotlib seaborn pandas numpy

# 2. Regenerate CHARIS feature cache (if needed)
python regen_cache.py

# 3. Run XGBoost pipeline (Paper 2)
python full_pipeline_qt.py

# 4. Run MIMIC-III validation
python mimic_validate.py

# 5. Run BiLSTM forecasting pipeline (Paper 3)
python -u bilstm_forecast.py

# 6. Add hardware subjects: drop any icp_{N}_{age}_{sex}.csv into hw-tests/
#    The pipeline auto-detects all CSVs — no code changes needed
```

---

## Methodology Highlights

### Why LOPO CV and Not Simple Train-Test Split?

In medical ML, patient-level leakage is the most common source of inflated results. If windows from the same patient appear in both train and test, the model memorises patient-specific features rather than learning generalisable ICP dynamics.

**Leave-One-Patient-Out CV** forces the model to predict on a patient it has *never seen in any form*:

```
Fold 1: Train [P2-P13] → Test [P1]     AUC: 0.9586
Fold 2: Train [P1,P3-P13] → Test [P2]  AUC: 0.9707
...
Fold 13: Train [P1-P12] → Test [P13]   AUC: 0.9509
─────────────────────────────────────────────────────
Mean LOPO AUC: 0.9611 ± 0.058
```

Each fold also fits its own QuantileTransformer — no distribution information leaks from future patients into past folds.

### Why QuantileTransformer (Not Z-score)?

ICP features are highly skewed (most windows have low cardiac amplitude; pathological spikes create extreme outliers). Z-score normalisation is sensitive to these outliers. QT maps each feature to a standard normal distribution, making XGBoost more robust to the distribution shift between CHARIS (ICP waveform) and hardware (optical TM sensor).

---

## Limitations & Future Work

| Limitation | Impact | Path Forward |
|---|---|---|
| No simultaneous ear sensor + ICP bolt | Hardware-to-model domain gap unquantified | Clinical collaboration (PGI Chandigarh) |
| 16 hardware subjects | Pilot-scale hardware validation | Collect 100 subjects (ongoing) |
| CHARIS features from ICP waveform | Model trained on invasive signal | Hybrid training with hardware normals |
| BiLSTM tested on CPU only | Long training time | GPU cluster / Google Colab |

---
</div>
