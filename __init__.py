"""
Super Weights Research Framework

A modular framework for detecting, analyzing, and understanding super weights
in Large Language Models. Super weights are individual parameters that have
disproportionate impact on model performance.
"""

from .research.researcher import SuperWeightResearchSession, quick_super_weight_analysis, compare_super_weights_across_models
from .detection.detector import SuperWeightDetector
from .detection.super_weight import SuperWeight
from .management.manager import SuperWeightManager
from .analysis.analyzer import SuperWeightAnalyzer
from .utils.datasets import DatasetLoader

__version__ = "0.1.0"
__all__ = [
    # Main research interface
    "SuperWeightResearchSession",
    "quick_super_weight_analysis", 
    "compare_super_weights_across_models",
    
    # Core components
    "SuperWeightDetector",
    "SuperWeight", 
    "SuperWeightManager",
    "SuperWeightAnalyzer",
    
    # Utilities
    "DatasetLoader"
]

