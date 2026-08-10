"""Arterial Stiffness Index calculator"""

import numpy as np
from scipy.signal import find_peaks


class ASICalculator:
    @staticmethod
    def from_abp(abp_window):
        """Calculate ASI from ABP waveform"""
        peaks, _ = find_peaks(abp_window, distance=20)
        troughs, _ = find_peaks(-abp_window, distance=20)
        
        if len(peaks) > 1 and len(troughs) > 1:
            sbp_beats = []
            dbp_beats = []
            for peak_idx in peaks:
                preceding = troughs[troughs < peak_idx]
                if len(preceding) > 0:
                    sbp_beats.append(abp_window[peak_idx])
                    dbp_beats.append(abp_window[preceding[-1]])
            
            if len(sbp_beats) > 1:
                slope = np.polyfit(sbp_beats, dbp_beats, 1)[0]
                return float(np.clip(1 - slope, 0, 1))
        
        return 0.5
