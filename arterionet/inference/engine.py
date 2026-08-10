"""
InferenceEngine: Smart device selection (PyTorch/MLX) for Y-NET ABP reconstruction
"""

import numpy as np
import torch
import platform
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Unified inference engine supporting both PyTorch and MLX backends.
    
    Auto-selects optimal device:
    - MLX on Apple Silicon (M1/M2/M3) — 4-5× faster
    - CUDA on NVIDIA GPUs
    - CPU as fallback
    """
    
    def __init__(self, model_path: str, device: Optional[str] = "auto"):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to Y-NET checkpoint (.pth)
            device: "auto", "mlx", "cuda", or "cpu"
        """
        self.model_path = model_path
        
        if device == "auto":
            self.device = self._detect_best_device()
        else:
            self.device = device
        
        logger.info(f"Inference device: {self.device}")
        
        # Load model on selected device
        if self.device == "mlx":
            self._load_mlx()
        else:
            self._load_torch()
    
    def _detect_best_device(self) -> str:
        """Auto-detect optimal device for inference."""
        if self._has_apple_silicon():
            try:
                import mlx.core as mx
                return "mlx"
            except ImportError:
                logger.warning("MLX not installed. Falling back to PyTorch MPS.")
                return "mps" if torch.backends.mps.is_available() else "cpu"
        
        elif torch.cuda.is_available():
            return "cuda"
        
        return "cpu"
    
    def _has_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon (M1/M2/M3)."""
        return (
            platform.processor() == 'arm' 
            or 'arm64' in platform.machine()
        ) and platform.system() == 'Darwin'
    
    def _load_torch(self):
        """Load Y-NET model in PyTorch."""
        from arterionet.inference.y_net_torch import YNETModelTorch
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.model = YNETModelTorch()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Y-NET model loaded on {self.device}")
    
    def _load_mlx(self):
        """Load Y-NET model in MLX."""
        try:
            from arterionet.inference.y_net_mlx import YNETModelMLX
            
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            self.model = YNETModelMLX.from_torch(checkpoint)
            logger.info("Y-NET model loaded in MLX (Apple Silicon optimized)")
        
        except ImportError:
            logger.error("MLX not available. Install with: pip install arterionet[mlx]")
            raise
    
    def predict(
        self, 
        ecg: np.ndarray, 
        ppg: np.ndarray,
        return_markers: bool = True
    ) -> dict:
        """
        Reconstruct ABP waveform and extract biomarkers.
        
        Args:
            ecg: ECG signal (250 samples, normalized)
            ppg: PPG signal (250 samples, normalized)
            return_markers: Include ASI/BPV in output
        
        Returns:
            dict with keys: 'abp', 'asi', 'bpv', 'r_delta'
        """
        if self.device == "mlx":
            return self._predict_mlx(ecg, ppg, return_markers)
        else:
            return self._predict_torch(ecg, ppg, return_markers)
    
    def _predict_torch(
        self, 
        ecg: np.ndarray, 
        ppg: np.ndarray,
        return_markers: bool = True
    ) -> dict:
        """PyTorch inference."""
        with torch.no_grad():
            ecg_t = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0).to(self.device)
            ppg_t = torch.tensor(ppg, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            abp, asi, bpv = self.model(ecg_t, ppg_t)
            
            abp_np = abp.cpu().numpy().squeeze()
            asi_np = asi.cpu().numpy().squeeze()
            bpv_np = bpv.cpu().numpy().squeeze()
        
        result = {
            'abp': abp_np,
            'asi': float(asi_np),
            'bpv': bpv_np,
            'device': self.device,
        }
        
        return result
    
    def _predict_mlx(
        self, 
        ecg: np.ndarray, 
        ppg: np.ndarray,
        return_markers: bool = True
    ) -> dict:
        """MLX inference (Apple Silicon native)."""
        try:
            import mlx.core as mx
        except ImportError:
            raise ImportError("MLX required for this inference. Install: pip install mlx")
        
        ecg_mx = mx.array(ecg.astype(np.float32))
        ppg_mx = mx.array(ppg.astype(np.float32))
        
        abp, asi, bpv = self.model(ecg_mx, ppg_mx)
        
        # Evaluate lazy arrays
        mx.eval(abp, asi, bpv)
        
        abp_np = np.array(abp).squeeze()
        asi_np = float(np.array(asi).squeeze())
        bpv_np = np.array(bpv).squeeze()
        
        result = {
            'abp': abp_np,
            'asi': asi_np,
            'bpv': bpv_np,
            'device': 'mlx',
        }
        
        return result
    
    def batch_predict(
        self, 
        ecg_batch: np.ndarray, 
        ppg_batch: np.ndarray,
        batch_size: int = 32
    ) -> list:
        """
        Process multiple ECG/PPG pairs.
        
        Args:
            ecg_batch: (N, 250) ECG signals
            ppg_batch: (N, 250) PPG signals
            batch_size: Process N samples at once
        
        Returns:
            List of prediction dicts
        """
        n_samples = len(ecg_batch)
        results = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            
            for j in range(i, end_idx):
                result = self.predict(ecg_batch[j], ppg_batch[j])
                results.append(result)
        
        return results
