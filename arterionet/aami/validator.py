"""
AAMI/ISO 81060-3:2022 Accuracy Criteria Validator
"""

import numpy as np
from scipy.stats import pearsonr
from typing import Dict


class AAMIValidator:
    """
    ISO 81060-3:2022 Blood Pressure Monitor Accuracy Criteria.
    
    Criteria:
    - Mean Error (ME): ≤ ±6.0 mmHg
    - Standard Deviation (SD): ≤ 10.0 mmHg
    """
    
    CRITERIA = {
        "mean_error_max_mmhg": 6.0,
        "std_dev_max_mmhg": 10.0,
    }
    
    @staticmethod
    def validate(sbp_pred: np.ndarray, sbp_true: np.ndarray, 
                 dbp_pred: np.ndarray, dbp_true: np.ndarray) -> dict:
        """
        Check compliance with AAMI criteria.
        
        Args:
            sbp_pred: Predicted SBP values (mmHg)
            sbp_true: Ground truth SBP (mmHg)
            dbp_pred: Predicted DBP values (mmHg)
            dbp_true: Ground truth DBP (mmHg)
        
        Returns:
            dict with compliance status for SBP/DBP
        """
        sbp_error = sbp_pred - sbp_true
        dbp_error = dbp_pred - dbp_true
        
        sbp_me = np.mean(sbp_error)
        sbp_sd = np.std(sbp_error, ddof=1)
        dbp_me = np.mean(dbp_error)
        dbp_sd = np.std(dbp_error, ddof=1)
        
        sbp_r, _ = pearsonr(sbp_pred, sbp_true)
        dbp_r, _ = pearsonr(dbp_pred, dbp_true)
        
        return {
            "sbp": {
                "mean_error": float(sbp_me),
                "std_dev": float(sbp_sd),
                "pearson_r": float(sbp_r),
                "me_pass": abs(sbp_me) <= AAMIValidator.CRITERIA["mean_error_max_mmhg"],
                "sd_pass": sbp_sd <= AAMIValidator.CRITERIA["std_dev_max_mmhg"],
            },
            "dbp": {
                "mean_error": float(dbp_me),
                "std_dev": float(dbp_sd),
                "pearson_r": float(dbp_r),
                "me_pass": abs(dbp_me) <= AAMIValidator.CRITERIA["mean_error_max_mmhg"],
                "sd_pass": dbp_sd <= AAMIValidator.CRITERIA["std_dev_max_mmhg"],
            }
        }
    
    @staticmethod
    def grade(validation_result: dict) -> str:
        """
        Assign compliance grade (A/B/C).
        
        A: All criteria pass
        B: SD passes, ME marginal
        C: Needs improvement
        """
        sbp = validation_result["sbp"]
        dbp = validation_result["dbp"]
        
        if sbp["me_pass"] and sbp["sd_pass"] and dbp["me_pass"] and dbp["sd_pass"]:
            return "A"
        elif sbp["sd_pass"] and dbp["sd_pass"]:
            return "B"
        else:
            return "C"
