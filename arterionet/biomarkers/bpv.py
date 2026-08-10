"""Blood Pressure Variability calculator"""

import numpy as np
from scipy.signal import find_peaks


class BPVCalculator:
    @staticmethod
    def from_abp(abp_window):
        """Calculate BPV from ABP waveform"""
        peaks, _ = find_peaks(abp_window, distance=20)
        troughs, _ = find_peaks(-abp_window, distance=20)
        
        sbp_beats = []
        dbp_beats = []
        for peak_idx in peaks:
            preceding = troughs[troughs < peak_idx]
            if len(preceding) > 0:
                sbp_beats.append(abp_window[peak_idx])
                dbp_beats.append(abp_window[preceding[-1]])
        
        sbp_sd = np.std(sbp_beats, ddof=1) if len(sbp_beats) > 1 else 0.0
        dbp_sd = np.std(dbp_beats, ddof=1) if len(dbp_beats) > 1 else 0.0
        
        return float(sbp_sd), float(dbp_sd)
