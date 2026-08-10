"""
ArterioNet: Cuffless Arterial Blood Pressure Reconstruction from ECG and PPG

A deep learning package for reconstructing continuous arterial blood pressure waveforms
from dual-channel biosignals (ECG + PPG), with extraction of clinical biomarkers including
Arterial Stiffness Index (ASI) and Blood Pressure Variability (BPV).

Authors:
  - McGovern Twumasi Owusu-Bekoe
  - Emefa Abena Apedo

License: MIT
"""

__version__ = "0.1.0"
__authors__ = [
    "McGovern Twumasi Owusu-Bekoe",
    "Emefa Abena Apedo"
]

# Core inference API
from arterionet.inference.engine import InferenceEngine
from arterionet.inference.model_loader import ModelLoader

# Biomarkers
from arterionet.biomarkers.asi import ASICalculator
from arterionet.biomarkers.bpv import BPVCalculator

# AAMI validation
from arterionet.aami.validator import AAMIValidator


# Utilities
from arterionet.utils.preprocessing import ECGPreprocessor, PPGPreprocessor
from arterionet.utils.normalization import ABPNormalizer
from arterionet.utils.metrics import calculate_metrics, pearson_r

# Public API
__all__ = [
    # Core
    "InferenceEngine",
    "ModelLoader",
    
    # Biomarkers
    "ASICalculator",
    "BPVCalculator",
    "MorningSurgeCalculator",
    "NocturnalDippingCalculator",
    
    # AAMI
    "AAMIValidator",
    "AAMIReporter",
    
    # Utils
    "ECGPreprocessor",
    "PPGPreprocessor",
    "ABPNormalizer",
    "calculate_metrics",
    "pearson_r",
]

print(f"ArterioNet v{__version__} loaded successfully")
