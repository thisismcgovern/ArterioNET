"""
ModelLoader: Download and load pretrained Y-NET checkpoints
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load pretrained Y-NET models."""
    
    AVAILABLE_MODELS = {
        "ynet-v2-kachuee": {
            "url": "https://github.com/mcgovern-twumasi/arterionet/releases/download/v0.1.0/ynet_v2_kachuee.pth",
            "description": "Y-NET trained on Kachuee MIMIC-II dataset",
            "metrics": {
                "dbp_pass_aami": True,
                "waveform_r": 0.984,
                "sbp_me": -3.57,
                "dbp_me": -2.21,
            }
        },
        "ynet-v2-pulsedb": {
            "url": "https://github.com/mcgovern-twumasi/arterionet/releases/download/v0.1.0/ynet_v2_pulsedb.pth",
            "description": "Y-NET trained on PulseDB dataset (external validation)",
            "metrics": {
                "dbp_pass_aami": True,
                "waveform_r": 0.982,
            }
        },
    }
    
    @classmethod
    def list_models(cls):
        """List available pretrained models."""
        for model_id, info in cls.AVAILABLE_MODELS.items():
            print(f"\n{model_id}")
            print(f"  Description: {info['description']}")
            print(f"  Metrics: {info['metrics']}")
    
    @classmethod
    def load(cls, model_name: str = "ynet-v2-kachuee", cache_dir: str = None):
        """
        Load a pretrained model.
        
        Args:
            model_name: Model ID from available_models
            cache_dir: Directory to cache checkpoints (default: ~/.arterionet/models)
        
        Returns:
            Loaded model path
        """
        if model_name not in cls.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls.AVAILABLE_MODELS.keys())}")
        
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.arterionet/models")
        
        os.makedirs(cache_dir, exist_ok=True)
        
        model_path = os.path.join(cache_dir, f"{model_name}.pth")
        
        # If already cached, return path
        if os.path.exists(model_path):
            logger.info(f"Loaded cached model: {model_path}")
            return model_path
        
        # Download from GitHub
        url = cls.AVAILABLE_MODELS[model_name]["url"]
        logger.info(f"Downloading {model_name} from {url}...")
        
        try:
            import urllib.request
            urllib.request.urlretrieve(url, model_path)
            logger.info(f"Model saved to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise
