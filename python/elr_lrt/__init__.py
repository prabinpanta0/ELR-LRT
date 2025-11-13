# python/elr_lrt/__init__.py
"""
ELR-LRT: Efficient Low-Resource Latent Reasoning Transformer

A package for efficient transformer-based sequence processing with dynamic byte patching,
continuous latent reasoning, and reinforcement learning fine-tuning.
"""

from .dbpm import patch_sequence
from .model import ELRLRTModel

__all__ = ['patch_sequence', 'ELRLRTModel']
__version__ = '0.1.6'