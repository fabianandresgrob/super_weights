#!/usr/bin/env python3
"""
Vocabulary Analysis Script for Super Weights

This script runs comprehensive vocabulary analyses from VocabularyAnalyzer on one or more
HuggingFace models. It detects super weights, analyzes their vocabulary effects, measures
interventional impacts, and generates machine-readable logs with visualizations.

Features:
- Detect super weights using configurable thresholds
- Analyze neuron vocabulary effects with optional cosine similarity
- Measure interventional impacts on loss, entropy, top-k margin, stopword mass
- Generate cascade effect analysis across layers
- Token class enrichment analysis
- Control experiments for validation
- Bootstrap confidence intervals
- Reproducible results with fixed seeds
- GPU-aware memory management
"""

import argparse
import json
import logging
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple
import csv

import numpy as np
import torch
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

# Ensure reproducibility
matplotlib.use('Agg')  # Non-interactive backend
plt.style.use('default')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from research.researcher import SuperWeightResearchSession
from analysis.vocabulary import VocabularyAnalyzer
from utils.datasets import DatasetLoader
from detection.super_weight import SuperWeight, MoESuperWeight

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


class VocabularyAnalysisRunner:
    """Main class for running comprehensive vocabulary analyses"""
    
    def __init__(self, 
                 output_dir: str = "results/vocab-analysis",
                 seed: int = 42,
                 log_level: int = logging.INFO):
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.log_level = log_level
        
        # Set seeds for reproducibility
        self._set_seeds()
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Create base directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset loader
        self.dataset_loader = DatasetLoader(seed=seed)
        
        # Track memory usage
        self.device_info = self._get_device_info()
        self.logger.info(f"Device info: {self.device_info}")
        
    def _get_device_info(self) -> Dict[str, Any]:
        """Get device information for logging"""
        device_info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
        }
        
        if torch.cuda.is_available():
            device_info.update({
                'device_name': torch.cuda.get_device_name(),
                'memory_allocated': torch.cuda.memory_allocated(),
                'memory_reserved': torch.cuda.memory_reserved()
            })
        
        return device_info
        
    def _set_seeds(self):
        """Set all random seeds for reproducibility"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        
    def _setup_logger(self, model_name: Optional[str] = None) -> logging.Logger:
        """Setup logging configuration"""
        logger_name = f"VocabAnalysis_{model_name}" if model_name else "VocabAnalysis"
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.log_level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if model-specific
        if model_name:
            log_file = self.output_dir / f"{self._sanitize_model_name(model_name)}" / "analysis.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _sanitize_model_name(self, model_name: str) -> str:
        """Sanitize model name for filesystem use"""
        return model_name.replace('/', '-').replace('\\', '-')
    
    def _get_model_specific_spike_threshold(self, model_name: str, user_threshold: Optional[float] = None) -> float:
        """Get model-specific spike threshold based on model architecture"""
        
        # If user provided a threshold, use it
        if user_threshold is not None:
            return user_threshold
        
        # Model-specific thresholds based on architecture
        model_thresholds = {
            'llama': 120.0,
            'phi': 150.0,
            'olmo': 70.0,
            'mistral': 100.0
        }
        
        # Default threshold
        default_threshold = 50.0
        
        # Parse model name to determine architecture
        model_lower = model_name.lower()
        
        for arch_name, threshold in model_thresholds.items():
            if arch_name in model_lower:
                return threshold
        
        # Return default if no match found
        return default_threshold
    
    def run_analysis(self, 
                    model_names: List[str],
                    dataset: str,
                    tokens: int,
                    window_len: int,
                    stride: int,
                    drop_first_token: bool,
                    enable_cosine: bool,
                    enable_cascade: bool,
                    enable_enrichment: bool,
                    bootstrap: int,
                    detection_config: Dict[str, Any],
                    cache_dir: str = '~/models/') -> Dict[str, Any]:
        """
        Run vocabulary analysis on multiple models
        
        Args:
            model_names: List of model names or paths
            dataset: Dataset name for evaluation
            tokens: Total tokens to evaluate
            window_len: Window length for sliding windows
            stride: Stride between windows
            drop_first_token: Whether to drop first token in each window
            enable_cosine: Whether to compute cosine variant
            enable_cascade: Whether to enable cascade analysis
            enable_enrichment: Whether to enable token class enrichment
            bootstrap: Number of bootstrap samples (0 to disable)
            detection_config: Configuration for super weight detection
            cache_dir: Directory to cache downloaded models
            
        Returns:
            Dictionary with analysis results for all models
        """
        self.logger.info("Starting vocabulary analysis")
        self.logger.info(f"Models: {model_names}")
        self.logger.info(f"Dataset: {dataset}, Tokens: {tokens}")
        self.logger.info(f"Window: {window_len}, Stride: {stride}, Drop first: {drop_first_token}")
        self.logger.info(f"Features: cosine={enable_cosine}, cascade={enable_cascade}, enrichment={enable_enrichment}")
        self.logger.info(f"Bootstrap samples: {bootstrap}")
        
        all_results = {
            'config': {
                'models': model_names,
                'dataset': dataset,
                'tokens': tokens,
                'window_len': window_len,
                'stride': stride,
                'drop_first_token': drop_first_token,
                'enable_cosine': enable_cosine,
                'enable_cascade': enable_cascade,
                'enable_enrichment': enable_enrichment,
                'bootstrap': bootstrap,
                'detection_config': detection_config,
                'seed': self.seed,
                'timestamp': datetime.now().isoformat()
            },
            'model_results': {},
            'cross_model_summary': {}
        }
        
        # Analyze each model
        for model_name in model_names:
            try:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"ANALYZING MODEL: {model_name}")
                self.logger.info(f"{'='*60}")
                
                model_results = self._analyze_single_model(
                    model_name=model_name,
                    dataset=dataset,
                    tokens=tokens,
                    window_len=window_len,
                    stride=stride,
                    drop_first_token=drop_first_token,
                    enable_cosine=enable_cosine,
                    enable_cascade=enable_cascade,
                    enable_enrichment=enable_enrichment,
                    bootstrap=bootstrap,
                    detection_config=detection_config,
                    cache_dir=cache_dir
                )
                
                all_results['model_results'][model_name] = model_results
                
                # Save model-specific results
                self._save_model_results(model_name, model_results)
                
            except Exception as e:
                self.logger.error(f"Failed to analyze model {model_name}: {e}")
                all_results['model_results'][model_name] = {'error': str(e)}
                continue
        
        # Cross-model analysis
        if len(all_results['model_results']) > 1:
            all_results['cross_model_summary'] = self._cross_model_analysis(all_results['model_results'])
        
        # Save global config and summary
        self._save_global_results(all_results)
        
        # Generate summary README
        self._generate_summary_readme(all_results)
        
        return all_results
    
    def _analyze_single_model(self, 
                            model_name: str,
                            dataset: str,
                            tokens: int,
                            window_len: int,
                            stride: int,
                            drop_first_token: bool,
                            enable_cosine: bool,
                            enable_cascade: bool,
                            enable_enrichment: bool,
                            bootstrap: int,
                            detection_config: Dict[str, Any],
                            cache_dir: str = '~/models/') -> Dict[str, Any]:
        """Analyze a single model"""
        
        # Setup model-specific directories
        model_dir = self._setup_model_directories(model_name)
        
        # Update logger for this model
        self.logger = self._setup_logger(model_name)
        
        # Load model and detect super weights
        self.logger.info(f"Loading model: {model_name}")
        with SuperWeightResearchSession.from_model_name(
            model_name, 
            cache_dir=cache_dir,
            log_level=self.log_level
        ) as session:
            
            # Create model-specific detection configuration
            model_detection_config = detection_config.copy()
            
            # Set model-specific spike threshold
            model_specific_threshold = self._get_model_specific_spike_threshold(
                model_name, 
                detection_config.get('spike_threshold')
            )
            model_detection_config['spike_threshold'] = model_specific_threshold
            
            # Log the threshold being used
            if detection_config.get('spike_threshold') is not None:
                self.logger.info(f"Using user-specified spike threshold: {model_specific_threshold}")
            else:
                self.logger.info(f"Using model-specific spike threshold for {model_name}: {model_specific_threshold}")
            
            # Detect super weights
            self.logger.info("Detecting super weights...")
            super_weights = session.detect_super_weights(**model_detection_config)
            
            self.logger.info(f"Detected {len(super_weights)} super weights")
            
            if not super_weights:
                self.logger.warning("No super weights detected, skipping analysis")
                return {'super_weights': [], 'neuron_analyses': [], 'super_weight_analyses': []}
            
            # Get vocabulary analyzer
            vocab_analyzer = VocabularyAnalyzer(
                session.model, 
                session.tokenizer, 
                session.manager, 
                session.mlp_handler
            )
            
            # Build held-out windows
            self.logger.info(f"Building evaluation windows from {dataset}")
            test_texts = self._load_evaluation_texts(dataset, tokens, window_len, stride, drop_first_token, session.tokenizer)
            
            # Analyze neurons (vocabulary effects)
            self.logger.info("Analyzing neuron vocabulary effects...")
            neuron_analyses = []
            for i, sw in enumerate(tqdm(super_weights, desc="Analyzing neurons")):
                try:
                    analysis = self._analyze_neuron_vocabulary_effects(
                        vocab_analyzer, sw, enable_cosine, enable_enrichment
                    )
                    neuron_analyses.append(analysis)
                    
                    # Save individual neuron card
                    self._save_neuron_card(model_name, analysis, model_dir)
                    
                    # Generate neuron plot
                    if enable_cosine or True:  # Always generate basic plots
                        self._generate_neuron_plot(model_name, analysis, model_dir)
                        
                except Exception as e:
                    self.logger.error(f"Failed to analyze neuron for SW {i}: {e}")
                    continue
            
            # Analyze super weight interventions
            self.logger.info("Analyzing super weight interventions...")
            super_weight_analyses = []
            for i, sw in enumerate(tqdm(super_weights, desc="Analyzing interventions")):
                try:
                    analysis = self._analyze_super_weight_intervention(
                        vocab_analyzer, sw, test_texts, enable_cascade, bootstrap
                    )
                    super_weight_analyses.append(analysis)
                    
                    # Save individual super weight card
                    self._save_super_weight_card(model_name, analysis, model_dir)
                    
                    # Generate super weight plots
                    self._generate_super_weight_plots(model_name, analysis, model_dir)
                    
                except Exception as e:
                    self.logger.error(f"Failed to analyze super weight {i}: {e}")
                    continue
            
            # Generate CSV summaries
            self._generate_csv_summaries(model_name, neuron_analyses, super_weight_analyses, model_dir)
            
            return {
                'model_name': model_name,
                'super_weights': [self._serialize_super_weight(sw) for sw in super_weights],
                'neuron_analyses': neuron_analyses,
                'super_weight_analyses': super_weight_analyses,
                'detection_config': model_detection_config,
                'analysis_summary': {
                    'total_super_weights': len(super_weights),
                    'successful_neuron_analyses': len(neuron_analyses),
                    'successful_intervention_analyses': len(super_weight_analyses)
                }
            }
    
    def _load_evaluation_texts(self, dataset: str, tokens: int, window_len: int, stride: int, 
                     drop_first_token: bool, tokenizer) -> List[str]:
        """Load evaluation texts with sliding windows and validate token requirements"""
        self.logger.info(f"Loading {dataset} dataset...")
        
        # Calculate minimum tokens needed for at least one window
        min_tokens_needed = window_len + stride  # Need at least one full window plus stride
        if tokens < min_tokens_needed:
            self.logger.warning(f"Requested tokens ({tokens}) < minimum needed ({min_tokens_needed}). Adjusting to {min_tokens_needed}")
            tokens = min_tokens_needed

        if dataset == 'wikitext-2':
            # Estimate how many samples we need to get the requested tokens
            # Based on empirical observations of WikiText-2:
            # - Average text length after filtering (min_length=50): ~200 tokens
            # - With sliding windows (stride=512, window=1024), we get ~1.5 windows per 1024 tokens of text
            # - Account for overhead and variability with a safety factor
            
            avg_tokens_per_text = 200  # Conservative estimate for WikiText-2 after filtering
            tokens_per_window = window_len - (1 if drop_first_token else 0)
            windows_per_text = max(1, (avg_tokens_per_text - window_len) // stride + 1)
            tokens_per_sample = windows_per_text * tokens_per_window
            
            # Calculate required samples with safety margins
            estimated_samples_needed = max(100, int(tokens / tokens_per_sample * 1.5))  # 50% safety margin
            
            # Cap at reasonable maximum to avoid memory issues
            max_samples = min(5000, estimated_samples_needed)
            
            self.logger.info(f"Requesting {tokens:,} tokens, estimated need {estimated_samples_needed} samples, using {max_samples}")
            
            # Load texts with computed upper bound
            raw_texts = self.dataset_loader.load_perplexity_dataset(
                dataset_name='wikitext',
                config='wikitext-2-raw-v1',
                split='test',
                n_samples=max_samples,
                min_length=50
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        # Convert to sliding windows and count tokens
        windowed_texts = []
        total_tokens_seen = 0
        
        for text in raw_texts:
            if total_tokens_seen >= tokens:
                break
                
            # Tokenize text
            text_tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Create sliding windows
            for start_idx in range(0, len(text_tokens) - window_len + 1, stride):
                if total_tokens_seen >= tokens:
                    break
                    
                window_tokens = text_tokens[start_idx:start_idx + window_len]
                
                # Drop first token if requested
                if drop_first_token and len(window_tokens) > 1:
                    window_tokens = window_tokens[1:]
            
                window_text = tokenizer.decode(window_tokens)
                windowed_texts.append(window_text)
                total_tokens_seen += len(window_tokens)

        # Validation check with better error message
        if len(windowed_texts) == 0:
            raise ValueError(f"Could not create any windows with tokens={tokens}, window_len={window_len}, stride={stride}")

        # Enhanced validation with specific recommendations
        token_efficiency = total_tokens_seen / tokens if tokens > 0 else 0
        if token_efficiency < 0.5:  # Got less than 50% of requested tokens
            self.logger.warning(f"Only got {total_tokens_seen:,} tokens out of {tokens:,} requested ({token_efficiency:.1%} efficiency)")
            if token_efficiency < 0.2:  # Very low efficiency
                self.logger.warning(f"Very low token efficiency! Consider:")
                self.logger.warning(f"  - Reducing --tokens to {total_tokens_seen}")
                self.logger.warning(f"  - Reducing --window-len from {window_len}")
                self.logger.warning(f"  - Reducing --stride from {stride}")
        elif token_efficiency < 0.8:
            self.logger.info(f"Got {total_tokens_seen:,} tokens out of {tokens:,} requested ({token_efficiency:.1%} efficiency)")
        else:
            self.logger.info(f"Successfully got {total_tokens_seen:,} tokens ({token_efficiency:.1%} of requested)")

        self.logger.info(f"Created {len(windowed_texts)} windows with {total_tokens_seen:,} total tokens")
        return windowed_texts
    
    def _analyze_neuron_vocabulary_effects(self, 
                                         vocab_analyzer: VocabularyAnalyzer,
                                         super_weight: SuperWeight,
                                         enable_cosine: bool,
                                         enable_enrichment: bool) -> Dict[str, Any]:
        """Analyze vocabulary effects for a neuron containing the super weight"""
        
        # Run vocabulary analysis
        vocab_results = vocab_analyzer.analyze_neuron_vocabulary_effects(
            super_weight, apply_universal_neurons_processing=True
        )
        
        # Check for errors
        if 'error' in vocab_results:
            raise Exception(vocab_results['error'])
        
        # Extract neuron information
        layer_idx = super_weight.layer
        neuron_idx = super_weight.row  # Assuming row represents neuron index
        
        # Build neuron card
        neuron_card = {
            'model': vocab_analyzer.model.name_or_path if hasattr(vocab_analyzer.model, 'name_or_path') else 'unknown',
            'layer': layer_idx,
            'neuron': neuron_idx,
            'tensor_name': super_weight.component,
            'coordinate': {'row': super_weight.row, 'col': super_weight.column},
            'original_value': float(super_weight.original_value) if super_weight.original_value is not None else None,
            'magnitude_product': super_weight.magnitude_product,
            'moments': {},
            'top_tokens_up': [],
            'top_tokens_down': [],
            'label': 'unknown',
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract moments from raw analysis
        if 'raw_analysis' in vocab_results and 'statistics' in vocab_results['raw_analysis']:
            stats = vocab_results['raw_analysis']['statistics']
            neuron_card['moments'] = {
                'var': float(stats.get('variance', 0.0)),
                'skew': float(stats.get('skew', 0.0)),
                'kurt': float(stats.get('kurtosis', 0.0))
            }
        
        # Extract cosine moments if enabled and available
        if enable_cosine and 'cosine_analysis' in vocab_results and 'statistics' in vocab_results['cosine_analysis']:
            stats_cos = vocab_results['cosine_analysis']['statistics']
            neuron_card['moments_cos'] = {
                'var': float(stats_cos.get('variance', 0.0)),
                'skew': float(stats_cos.get('skew', 0.0)),
                'kurt': float(stats_cos.get('kurtosis', 0.0))
            }
        
        # Extract top tokens - FIX: Use correct path
        if 'raw_analysis' in vocab_results and 'top_tokens' in vocab_results['raw_analysis']:
            top_tokens = vocab_results['raw_analysis']['top_tokens']
            
            # Convert to required format
            for token_info in top_tokens.get('top_boosted', [])[:20]:
                neuron_card['top_tokens_up'].append({
                    'id': int(token_info['token_id']),
                    'tok': token_info['raw_token'],
                    'str': token_info['token_str'],
                    'effect': float(token_info['effect_magnitude'])
                })
            
            for token_info in top_tokens.get('top_suppressed', [])[:20]:
                neuron_card['top_tokens_down'].append({
                    'id': int(token_info['token_id']),
                    'tok': token_info['raw_token'],
                    'str': token_info['token_str'], 
                    'effect': float(token_info['effect_magnitude'])
                })
    
        # Extract classification label - FIX: Use correct path
        if 'raw_analysis' in vocab_results and 'classification' in vocab_results['raw_analysis']:
            neuron_card['label'] = vocab_results['raw_analysis']['classification'].get('type', 'unknown')
    
        # Add enrichment analysis if enabled - FIX: Use correct path
        if enable_enrichment and 'raw_analysis' in vocab_results and 'enrichment' in vocab_results['raw_analysis']:
            enrichment_results = vocab_results['raw_analysis']['enrichment']
            enrichment_data = []
            
            for class_name, score in enrichment_results.get('enrichment_scores', {}).items():
                examples = []
                if 'class_effects' in enrichment_results and class_name in enrichment_results['class_effects']:
                    examples = enrichment_results['class_effects'][class_name].get('top_effects', [])[:5]
                
                enrichment_data.append({
                    'class': class_name,
                    'score': float(score),
                    'examples': examples
                })
            neuron_card['enrichment'] = enrichment_data
    
        return neuron_card
    
    def _analyze_super_weight_intervention(self,
                                         vocab_analyzer: VocabularyAnalyzer,
                                         super_weight: SuperWeight,
                                         test_texts: List[str],
                                         enable_cascade: bool,
                                         bootstrap: int) -> Dict[str, Any]:
        """Analyze interventional effects of a super weight"""
        
        # Run intervention analysis
        intervention_results = vocab_analyzer.analyze_super_weight_intervention(
            super_weight, test_texts=test_texts, n_samples=len(test_texts)
        )
        
        # Add interventional token analysis
        token_effects = self._analyze_interventional_token_effects(
            vocab_analyzer, super_weight, test_texts[0] if test_texts else "The company develops technology."
        )
        
        # Build super weight card
        super_weight_card = {
            'model': vocab_analyzer.model.name_or_path if hasattr(vocab_analyzer.model, 'name_or_path') else 'unknown',
            'tensor': super_weight.component,
            'layer': super_weight.layer,
            'row': super_weight.row,
            'col': super_weight.column,
            'baselines': {
                'loss': float(intervention_results['baseline_loss']),
                'entropy': float(intervention_results['baseline_entropy']),
                'topk_margin': float(intervention_results['baseline_topk_margin']),
                'stopword_mass': float(intervention_results['baseline_stopword_mass']) if intervention_results['baseline_stopword_mass'] is not None else None
            },
            'deltas': {
                'loss': float(intervention_results['delta_loss']),
                'entropy': float(intervention_results['delta_entropy']),
                'topk_margin': float(intervention_results['delta_topk_margin']),
                'stopword_mass': float(intervention_results['delta_stopword_mass']) if intervention_results['delta_stopword_mass'] is not None else None
            },
            'interventional_tokens': token_effects,  # ADD: Token-level intervention effects
            'controls': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Add control experiments
        try:
            # Run comprehensive control experiments
            control_results = vocab_analyzer.analyze_controls_and_baselines(
                super_weight, test_texts=test_texts, n_samples=len(test_texts)
            )
            
            # Extract random coordinate control
            if 'random_baseline' in control_results:
                random_baseline = control_results['random_baseline']
                super_weight_card['controls']['random_coord_delta'] = {
                    'loss': float(random_baseline.get('random_delta_loss', 0.0)),
                    'entropy': 0.0,  # Not available in this format
                    'topk_margin': 0.0,  # Not available in this format  
                    'stopword_mass': 0.0  # Not available in this format
                }
            
            # Extract neuron vs scalar comparison
            if 'neuron_vs_scalar' in control_results:
                neuron_scalar = control_results['neuron_vs_scalar']
                super_weight_card['controls']['neuron_zero_delta'] = {
                    'loss': float(neuron_scalar.get('neuron_delta_loss', 0.0)),
                    'entropy': float(neuron_scalar.get('neuron_delta_entropy', 0.0)),
                    'topk_margin': float(neuron_scalar.get('neuron_delta_topk_margin', 0.0))
                }
            
        except Exception as e:
            self.logger.warning(f"Control experiments failed: {e}")
        
        # Add cascade analysis if enabled
        if enable_cascade:
            try:
                cascade_results = vocab_analyzer.analyze_cascade_effects(
                    super_weight, 
                    input_text=test_texts[0] if test_texts else "Apple Inc. is a tech company.",
                    num_layers=min(8, len(vocab_analyzer.mlp_handler.layers) // 4),  # Sample layers
                    top_k_margin=10
                )
                
                if 'error' not in cascade_results:
                    layer_effects = cascade_results.get('layer_effects', {})
                    actual_layer_indices = cascade_results.get('actual_layer_indices', [])
                    
                    super_weight_card['cascade'] = {
                        'layers': actual_layer_indices,
                        'delta_entropy': [layer_effects[layer_idx]['entropy_change'] for layer_idx in actual_layer_indices],
                        'delta_margin': [layer_effects[layer_idx]['margin_change'] for layer_idx in actual_layer_indices]
                    }
                
            except Exception as e:
                self.logger.warning(f"Cascade analysis failed: {e}")
        
        # Add MoE-specific analysis if applicable
        if isinstance(super_weight, MoESuperWeight):
            super_weight_card['moe'] = {
                'expert_idx': super_weight.expert_idx,
                'p_active': float(super_weight.p_active),
                'co_spike_score': float(super_weight.co_spike_score)
            }
        
        # Add bootstrap confidence intervals if requested
        if bootstrap > 0:
            super_weight_card['bootstrap'] = self._compute_bootstrap_intervals(
                vocab_analyzer, super_weight, test_texts, bootstrap
            )
        
        return super_weight_card
    
    def _analyze_interventional_token_effects(self, 
                                        vocab_analyzer: VocabularyAnalyzer,
                                        super_weight: SuperWeight,
                                        test_text: str = "The company develops technology solutions.") -> Dict[str, Any]:
        """
        Analyze how super weight intervention affects specific token probabilities.
        
        This gives us token-level changes due to the intervention, complementing
        the neuron vocabulary analysis.
        """
        try:
            # Tokenize a representative text
            tokens = vocab_analyzer.tokenizer(test_text, return_tensors='pt').to(vocab_analyzer.model.device)
            
            # Get baseline and intervention logits for last token
            with torch.no_grad():
                # Baseline
                baseline_outputs = vocab_analyzer.model(**tokens)
                baseline_logits = baseline_outputs.logits[0, -1, :].cpu()
                baseline_probs = torch.softmax(baseline_logits, dim=-1)
                
                # With intervention
                with vocab_analyzer.manager.temporary_zero([super_weight]):
                    modified_outputs = vocab_analyzer.model(**tokens)
                    modified_logits = modified_outputs.logits[0, -1, :].cpu()
                    modified_probs = torch.softmax(modified_logits, dim=-1)
                
                # Compute probability changes
                prob_changes = modified_probs - baseline_probs
                
                # Get top changed tokens
                special_token_ids = set(getattr(vocab_analyzer.tokenizer, 'all_special_ids', []))
                
                # Top boosted tokens (most positive changes)
                top_boosted_indices = torch.topk(prob_changes, 50).indices  # Get extra to filter
                top_boosted = []
                for idx in top_boosted_indices:
                    token_id = idx.item()
                    if token_id in special_token_ids:
                        continue
                        
                    try:
                        raw_token = vocab_analyzer.tokenizer.convert_ids_to_tokens([token_id])[0]
                        token_str = vocab_analyzer.tokenizer.convert_tokens_to_string([raw_token])
                        
                        top_boosted.append({
                            'token_id': token_id,
                            'token_str': token_str,
                            'raw_token': raw_token,
                            'prob_change': float(prob_changes[idx]),
                            'baseline_prob': float(baseline_probs[idx]),
                            'modified_prob': float(modified_probs[idx])
                        })
                        
                        if len(top_boosted) >= 20:
                            break
                    except:
                        pass
                
                # Top suppressed tokens (most negative changes)
                top_suppressed_indices = torch.topk(-prob_changes, 50).indices
                top_suppressed = []
                for idx in top_suppressed_indices:
                    token_id = idx.item()
                    if token_id in special_token_ids:
                        continue
                        
                    try:
                        raw_token = vocab_analyzer.tokenizer.convert_ids_to_tokens([token_id])[0]
                        token_str = vocab_analyzer.tokenizer.convert_tokens_to_string([raw_token])
                        
                        top_suppressed.append({
                            'token_id': token_id,
                            'token_str': token_str,
                            'raw_token': raw_token,
                            'prob_change': float(prob_changes[idx]),
                            'baseline_prob': float(baseline_probs[idx]),
                            'modified_prob': float(modified_probs[idx])
                        })
                        
                        if len(top_suppressed) >= 20:
                            break
                    except:
                        pass
                
                return {
                    'input_text': test_text,
                    'top_boosted_tokens': top_boosted,
                    'top_suppressed_tokens': top_suppressed,
                    'total_prob_change': float(torch.sum(torch.abs(prob_changes))),
                    'max_prob_increase': float(torch.max(prob_changes)),
                    'max_prob_decrease': float(torch.min(prob_changes))
                }
                
        except Exception as e:
            return {'error': f"Interventional token analysis failed: {str(e)}"}
    
    def _compute_bootstrap_intervals(self,
                                   vocab_analyzer: VocabularyAnalyzer,
                                   super_weight: SuperWeight,
                                   test_texts: List[str],
                                   n_bootstrap: int) -> Dict[str, Any]:
        """Compute bootstrap confidence intervals for intervention effects"""
        
        self.logger.info(f"Computing bootstrap intervals with {n_bootstrap} samples...")
        
        bootstrap_deltas = {'loss': [], 'entropy': [], 'topk_margin': [], 'stopword_mass': []}
        
        for _ in tqdm(range(n_bootstrap), desc="Bootstrap", leave=False):
            # Resample texts with replacement
            resampled_texts = np.random.choice(test_texts, size=len(test_texts), replace=True).tolist()
            
            try:
                # Run intervention on resampled data
                boot_results = vocab_analyzer.analyze_super_weight_intervention(
                    super_weight, test_texts=resampled_texts, n_samples=len(resampled_texts)
                )
                
                bootstrap_deltas['loss'].append(float(boot_results['delta_loss']))
                bootstrap_deltas['entropy'].append(float(boot_results['delta_entropy']))
                bootstrap_deltas['topk_margin'].append(float(boot_results['delta_topk_margin']))
                if boot_results['delta_stopword_mass'] is not None:
                    bootstrap_deltas['stopword_mass'].append(float(boot_results['delta_stopword_mass']))
                    
            except Exception as e:
                self.logger.warning(f"Bootstrap sample failed: {e}")
                continue
        
        # Compute confidence intervals
        confidence_intervals = {}
        for metric, deltas in bootstrap_deltas.items():
            if deltas:
                confidence_intervals[metric] = {
                    'mean': float(np.mean(deltas)),
                    'std': float(np.std(deltas)),
                    'ci_2.5': float(np.percentile(deltas, 2.5)),
                    'ci_97.5': float(np.percentile(deltas, 97.5))
                }
        
        return confidence_intervals
    
    def _serialize_super_weight(self, sw: SuperWeight) -> Dict[str, Any]:
        """Serialize a super weight object to dictionary"""
        base_dict = {
            'tensor_name': sw.component,
            'layer': sw.layer,
            'row': sw.row,
            'col': sw.column,
            'magnitude': float(sw.original_value) if sw.original_value is not None else None
        }
        
        if isinstance(sw, MoESuperWeight):
            base_dict.update({
                'expert_id': sw.expert_id,
                'p_active': float(sw.p_active) if sw.p_active is not None else None,
                'co_spike_score': float(sw.co_spike_score) if sw.co_spike_score is not None else None
            })
        
        return base_dict
    
    def _setup_model_directories(self, model_name: str) -> Path:
        """Setup directory structure for a model"""
        sanitized_name = self._sanitize_model_name(model_name)
        model_dir = self.output_dir / sanitized_name
        
        # Create subdirectories
        (model_dir / "metrics" / "neurons").mkdir(parents=True, exist_ok=True)
        (model_dir / "metrics" / "super_weights").mkdir(parents=True, exist_ok=True)
        (model_dir / "plots" / "neurons").mkdir(parents=True, exist_ok=True)
        (model_dir / "plots" / "super_weights").mkdir(parents=True, exist_ok=True)
        (model_dir / "summaries").mkdir(parents=True, exist_ok=True)
        
        return model_dir
    
    def _save_neuron_card(self, model_name: str, neuron_analysis: Dict[str, Any], model_dir: Path):
        """Save neuron card to JSONL file"""
        neurons_file = model_dir / "metrics" / "neurons" / "neurons.jsonl"
        
        # Append to JSONL file
        with open(neurons_file, 'a') as f:
            f.write(json.dumps(neuron_analysis) + '\n')
    
    def _save_super_weight_card(self, model_name: str, sw_analysis: Dict[str, Any], model_dir: Path):
        """Save super weight card to JSONL file"""
        sw_file = model_dir / "metrics" / "super_weights" / "super_weights.jsonl"
        
        # Append to JSONL file
        with open(sw_file, 'a') as f:
            f.write(json.dumps(sw_analysis) + '\n')
    
    def _generate_neuron_plot(self, model_name: str, neuron_analysis: Dict[str, Any], model_dir: Path):
        """Generate visualization for neuron vocabulary effects"""
        
        layer = neuron_analysis['layer']
        neuron = neuron_analysis['neuron']
        
        # Check if we have token data
        if not neuron_analysis['top_tokens_up'] and not neuron_analysis['top_tokens_down']:
            self.logger.warning(f"No token data for neuron L{layer}N{neuron}, skipping plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f"Neuron Vocabulary Effects: {model_name} L{layer}N{neuron}", fontsize=14)
        
        # Plot 1: Top boosted tokens
        if neuron_analysis['top_tokens_up']:
            tokens_up = neuron_analysis['top_tokens_up'][:15]  # Top 15
            token_strs = [t['str'][:20] + ('...' if len(t['str']) > 20 else '') for t in tokens_up]
            effects_up = [t['effect'] for t in tokens_up]
            
            y_positions = range(len(token_strs))
            bars = axes[0].barh(y_positions, effects_up, color='green', alpha=0.7)
            axes[0].set_yticks(y_positions)
            axes[0].set_yticklabels(token_strs, fontsize=8)
            axes[0].set_xlabel('Vocabulary Effect')
            axes[0].set_title('Top Boosted Tokens')
            axes[0].grid(True, alpha=0.3)
            axes[0].invert_yaxis()  # Show highest at top
        else:
            axes[0].text(0.5, 0.5, 'No boosted tokens found', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Top Boosted Tokens (None Found)')
        
        # Plot 2: Top suppressed tokens
        if neuron_analysis['top_tokens_down']:
            tokens_down = neuron_analysis['top_tokens_down'][:15]  # Top 15
            token_strs = [t['str'][:20] + ('...' if len(t['str']) > 20 else '') for t in tokens_down]
            effects_down = [t['effect'] for t in tokens_down]
            
            y_positions = range(len(token_strs))
            bars = axes[1].barh(y_positions, effects_down, color='red', alpha=0.7)
            axes[1].set_yticks(y_positions)
            axes[1].set_yticklabels(token_strs, fontsize=8)
            axes[1].set_xlabel('Vocabulary Effect')
            axes[1].set_title('Top Suppressed Tokens')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = model_dir / "plots" / "neurons" / f"{layer}-{neuron}-effects.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_super_weight_plots(self, model_name: str, sw_analysis: Dict[str, Any], model_dir: Path):
        """Generate visualization for super weight interventional effects"""
        
        layer = sw_analysis['layer']
        row = sw_analysis['row']
        col = sw_analysis['col']
        tensor = sw_analysis['tensor']
        
        # Plot 1: Delta metrics comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        metrics = ['loss', 'entropy', 'topk_margin', 'stopword_mass']
        deltas = [sw_analysis['deltas'].get(m, 0.0) for m in metrics]
        
        # Filter out None values
        valid_metrics = []
        valid_deltas = []
        for m, d in zip(metrics, deltas):
            if d is not None and not (np.isnan(d) or np.isinf(d)):
                valid_metrics.append(m.replace('_', ' ').title())
                valid_deltas.append(d)
        
        if valid_deltas:
            colors = ['red' if d > 0 else 'blue' for d in valid_deltas]
            bars = ax.bar(valid_metrics, valid_deltas, color=colors, alpha=0.7)
            ax.set_ylabel('Delta Value')
            ax.set_title(f'Super Weight Intervention Effects: {tensor} L{layer}R{row}C{col}')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels on bars
            for bar, delta in zip(bars, valid_deltas):
                height = bar.get_height()
                ax.annotate(f'{delta:.4f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No valid delta values found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Super Weight Intervention Effects: {tensor} L{layer}R{row}C{col} (No Data)')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = model_dir / "plots" / "super_weights" / f"{tensor}-L{layer}-R{row}-C{col}-deltas.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Interventional token effects if available
        if 'interventional_tokens' in sw_analysis and 'error' not in sw_analysis['interventional_tokens']:
            token_data = sw_analysis['interventional_tokens']
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(f'Interventional Token Effects: {tensor} L{layer}R{row}C{col}', fontsize=14)
            
            # Top boosted tokens (probability increases)
            if token_data['top_boosted_tokens']:
                tokens_up = token_data['top_boosted_tokens'][:15]
                token_strs = [t['token_str'][:20] + ('...' if len(t['token_str']) > 20 else '') for t in tokens_up]
                prob_changes = [t['prob_change'] for t in tokens_up]
                
                y_positions = range(len(token_strs))
                bars = axes[0].barh(y_positions, prob_changes, color='green', alpha=0.7)
                axes[0].set_yticks(y_positions)
                axes[0].set_yticklabels(token_strs, fontsize=8)
                axes[0].set_xlabel('Probability Change')
                axes[0].set_title('Top Probability Increases (Interventional)')
                axes[0].grid(True, alpha=0.3)
                axes[0].invert_yaxis()
            
            # Top suppressed tokens (probability decreases)
            if token_data['top_suppressed_tokens']:
                tokens_down = token_data['top_suppressed_tokens'][:15]
                token_strs = [t['token_str'][:20] + ('...' if len(t['token_str']) > 20 else '') for t in tokens_down]
                prob_changes = [t['prob_change'] for t in tokens_down]
                
                y_positions = range(len(token_strs))
                bars = axes[1].barh(y_positions, prob_changes, color='red', alpha=0.7)
                axes[1].set_yticks(y_positions)
                axes[1].set_yticklabels(token_strs, fontsize=8)
                axes[1].set_xlabel('Probability Change')
                axes[1].set_title('Top Probability Decreases (Interventional)')
                axes[1].grid(True, alpha=0.3)
                axes[1].invert_yaxis()
            
            plt.tight_layout()
            
            # Save interventional token plot
            token_plot_path = model_dir / "plots" / "super_weights" / f"{tensor}-L{layer}-R{row}-C{col}-tokens.png"
            plt.savefig(token_plot_path, dpi=150, bbox_inches='tight')
            plt.close()

        # Plot 3: Cascade effects if available
        if 'cascade' in sw_analysis:
            cascade_data = sw_analysis['cascade']
            
            if cascade_data['layers'] and cascade_data['delta_entropy']:
                fig, axes = plt.subplots(2, 1, figsize=(10, 8))
                
                layers = cascade_data['layers']
                
                # Entropy cascade
                axes[0].plot(layers, cascade_data['delta_entropy'], 'b-o')
                axes[0].set_xlabel('Layer')
                axes[0].set_ylabel('Delta Entropy')
                axes[0].set_title('Cascade Effect: Entropy')
                axes[0].grid(True, alpha=0.3)
                
                # Margin cascade
                if cascade_data['delta_margin']:
                    axes[1].plot(layers, cascade_data['delta_margin'], 'r-o')
                    axes[1].set_xlabel('Layer')
                    axes[1].set_ylabel('Delta Top-K Margin')
                    axes[1].set_title('Cascade Effect: Top-K Margin')
                    axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save cascade plot
                cascade_plot_path = model_dir / "plots" / "super_weights" / f"{tensor}-L{layer}-R{row}-C{col}-cascade.png"
                plt.savefig(cascade_plot_path, dpi=150, bbox_inches='tight')
                plt.close()
    
    def _generate_csv_summaries(self, 
                              model_name: str,
                              neuron_analyses: List[Dict],
                              super_weight_analyses: List[Dict],
                              model_dir: Path):
        """Generate CSV summaries for quick filtering and analysis"""
        
        # Neurons CSV
        if neuron_analyses:
            neurons_csv = model_dir / "summaries" / "neurons.csv"
            with open(neurons_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'model', 'layer', 'neuron', 'tensor_name', 'row', 'col',
                    'var', 'skew', 'kurt', 'var_cos', 'skew_cos', 'kurt_cos',
                    'label', 'top_boost_token', 'top_boost_effect',
                    'top_suppress_token', 'top_suppress_effect'
                ])
                writer.writeheader()
                
                for analysis in neuron_analyses:
                    row_data = {
                        'model': analysis['model'],
                        'layer': analysis['layer'],
                        'neuron': analysis['neuron'],
                        'tensor_name': analysis.get('tensor_name', ''),
                        'row': analysis['coordinate']['row'],
                        'col': analysis['coordinate']['col'],
                        'var': analysis['moments'].get('var', ''),
                        'skew': analysis['moments'].get('skew', ''),
                        'kurt': analysis['moments'].get('kurt', ''),
                        'var_cos': analysis.get('moments_cos', {}).get('var', ''),
                        'skew_cos': analysis.get('moments_cos', {}).get('skew', ''),
                        'kurt_cos': analysis.get('moments_cos', {}).get('kurt', ''),
                        'label': analysis['label'],
                        'top_boost_token': analysis['top_tokens_up'][0]['str'] if analysis['top_tokens_up'] else '',
                        'top_boost_effect': analysis['top_tokens_up'][0]['effect'] if analysis['top_tokens_up'] else '',
                        'top_suppress_token': analysis['top_tokens_down'][0]['str'] if analysis['top_tokens_down'] else '',
                        'top_suppress_effect': analysis['top_tokens_down'][0]['effect'] if analysis['top_tokens_down'] else ''
                    }
                    writer.writerow(row_data)
        
        # Super weights CSV
        if super_weight_analyses:
            sw_csv = model_dir / "summaries" / "super_weights.csv"
            with open(sw_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'model', 'tensor', 'layer', 'row', 'col',
                    'baseline_loss', 'baseline_entropy', 'baseline_topk_margin', 'baseline_stopword_mass',
                    'delta_loss', 'delta_entropy', 'delta_topk_margin', 'delta_stopword_mass',
                    'random_delta_loss', 'random_delta_entropy', 'neuron_delta_loss', 'neuron_delta_entropy',
                    'has_cascade', 'is_moe', 'p_active'
                ])
                writer.writeheader()
                
                for analysis in super_weight_analyses:
                    row_data = {
                        'model': analysis['model'],
                        'tensor': analysis['tensor'],
                        'layer': analysis['layer'],
                        'row': analysis['row'],
                        'col': analysis['col'],
                        'baseline_loss': analysis['baselines']['loss'],
                        'baseline_entropy': analysis['baselines']['entropy'],
                        'baseline_topk_margin': analysis['baselines']['topk_margin'],
                        'baseline_stopword_mass': analysis['baselines']['stopword_mass'],
                        'delta_loss': analysis['deltas']['loss'],
                        'delta_entropy': analysis['deltas']['entropy'],
                        'delta_topk_margin': analysis['deltas']['topk_margin'],
                        'delta_stopword_mass': analysis['deltas']['stopword_mass'],
                        'random_delta_loss': analysis['controls'].get('random_coord_delta', {}).get('loss', ''),
                        'random_delta_entropy': analysis['controls'].get('random_coord_delta', {}).get('entropy', ''),
                        'neuron_delta_loss': analysis['controls'].get('neuron_zero_delta', {}).get('loss', ''),
                        'neuron_delta_entropy': analysis['controls'].get('neuron_zero_delta', {}).get('entropy', ''),
                        'has_cascade': 'cascade' in analysis,
                        'is_moe': 'moe' in analysis,
                        'p_active': analysis.get('moe', {}).get('p_active', '')
                    }
                    writer.writerow(row_data)
    
    def _cross_model_analysis(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Perform cross-model analysis"""
        
        summary = {
            'total_models': len(model_results),
            'models_analyzed': list(model_results.keys()),
            'per_model_stats': {},
            'aggregate_stats': {}
        }
        
        # Per-model statistics
        for model_name, results in model_results.items():
            if 'error' in results:
                summary['per_model_stats'][model_name] = {'error': results['error']}
                continue
                
            summary['per_model_stats'][model_name] = {
                'super_weights_detected': len(results.get('super_weights', [])),
                'neurons_analyzed': len(results.get('neuron_analyses', [])),
                'interventions_analyzed': len(results.get('super_weight_analyses', []))
            }
        
        # Aggregate statistics
        valid_results = [r for r in model_results.values() if 'error' not in r]
        if valid_results:
            total_sw = sum(len(r.get('super_weights', [])) for r in valid_results)
            total_neurons = sum(len(r.get('neuron_analyses', [])) for r in valid_results)
            total_interventions = sum(len(r.get('super_weight_analyses', [])) for r in valid_results)
            
            summary['aggregate_stats'] = {
                'total_super_weights': total_sw,
                'total_neurons_analyzed': total_neurons,
                'total_interventions_analyzed': total_interventions,
                'avg_super_weights_per_model': total_sw / len(valid_results) if valid_results else 0
            }
        
        return summary
    
    def _save_model_results(self, model_name: str, results: Dict[str, Any]):
        """Save complete model results"""
        sanitized_name = self._sanitize_model_name(model_name)
        model_dir = self.output_dir / sanitized_name
        
        # Save run config
        config_file = model_dir / "run_config.json"
        with open(config_file, 'w') as f:
            json.dump(results.get('detection_config', {}), f, indent=2)
    
    def _save_global_results(self, all_results: Dict[str, Any]):
        """Save global analysis results and configuration"""
        
        # Save global config
        config_file = self.output_dir / "run_config.json"
        with open(config_file, 'w') as f:
            json.dump(all_results['config'], f, indent=2)
        
        # Save cross-model summary
        if all_results['cross_model_summary']:
            summary_file = self.output_dir / "cross_model_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(all_results['cross_model_summary'], f, indent=2)
    
    def _generate_summary_readme(self, all_results: Dict[str, Any]):
        """Generate summary README with analysis overview"""
        
        readme_content = f"""# Vocabulary Analysis Results

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration

- **Models**: {', '.join(all_results['config']['models'])}
- **Dataset**: {all_results['config']['dataset']}
- **Total Tokens**: {all_results['config']['tokens']:,}
- **Window Length**: {all_results['config']['window_len']}
- **Stride**: {all_results['config']['stride']}
- **Features Enabled**: 
  - Cosine similarity: {all_results['config']['enable_cosine']}
  - Cascade analysis: {all_results['config']['enable_cascade']}
  - Token enrichment: {all_results['config']['enable_enrichment']}
- **Bootstrap Samples**: {all_results['config']['bootstrap']}
- **Seed**: {all_results['config']['seed']}

## Results Summary

"""
        
        # Add per-model summaries
        for model_name, results in all_results['model_results'].items():
            if 'error' in results:
                readme_content += f"### {model_name}\n\n**ERROR**: {results['error']}\n\n"
                continue
                
            sanitized_name = self._sanitize_model_name(model_name)
            summary = results.get('analysis_summary', {})
            
            readme_content += f"""### {model_name}

- **Super Weights Detected**: {summary.get('total_super_weights', 0)}
- **Neurons Analyzed**: {summary.get('successful_neuron_analyses', 0)}
- **Interventions Analyzed**: {summary.get('successful_intervention_analyses', 0)}

**Files**:
- Neuron cards: `{sanitized_name}/metrics/neurons/neurons.jsonl`
- Super weight cards: `{sanitized_name}/metrics/super_weights/super_weights.jsonl`
- CSV summaries: `{sanitized_name}/summaries/`
- Plots: `{sanitized_name}/plots/`

"""
        
        # Add cross-model summary if available
        if 'cross_model_summary' in all_results and all_results['cross_model_summary']:
            cross_summary = all_results['cross_model_summary']
            readme_content += f"""## Cross-Model Summary

- **Total Models Analyzed**: {cross_summary.get('total_models', 0)}
- **Total Super Weights**: {cross_summary.get('aggregate_stats', {}).get('total_super_weights', 0)}
- **Average Super Weights per Model**: {cross_summary.get('aggregate_stats', {}).get('avg_super_weights_per_model', 0):.2f}

"""
        
        readme_content += """## Usage

To load and analyze results:

```python
import json
import pandas as pd

# Load neuron analyses
with open('<model>/metrics/neurons/neurons.jsonl', 'r') as f:
    neurons = [json.loads(line) for line in f]

# Load super weight analyses  
with open('<model>/metrics/super_weights/super_weights.jsonl', 'r') as f:
    super_weights = [json.loads(line) for line in f]

# Load CSV summaries
neurons_df = pd.read_csv('<model>/summaries/neurons.csv')
sw_df = pd.read_csv('<model>/summaries/super_weights.csv')
```

## File Formats

### Neuron Card Schema
```json
{
  "model": "...",
  "layer": 10, "neuron": 523,
  "moments": {"var": ..., "skew": ..., "kurt": ...},
  "moments_cos": {"var": ..., "skew": ..., "kurt": ...},
  "top_tokens_up": [{"id": 123, "tok": "Ġthe", "str": " the", "effect": 1.23}],
  "top_tokens_down": [...],
  "enrichment": [{"class": "digits_years", "score": 0.42, "examples": ["1999","2020"]}],
  "label": "prediction"
}
```

### Super Weight Card Schema
```json
{
  "model": "...",
  "tensor": "mlp_out", "layer": 10, "row": 523, "col": 768,
  "baselines": {"loss": ..., "entropy": ..., "topk_margin": ..., "stopword_mass": ...},
  "deltas": {"loss": +0.018, "entropy": +0.012, "topk_margin": -0.007, "stopword_mass": -0.004},
  "controls": {...},
  "cascade": {...}
}
```
"""
        
        # Save README
        readme_file = self.output_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)


def parse_arguments():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description="Vocabulary Analysis for Super Weights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input configuration
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        required=True,
        help='Model names or local paths (HF style)'
    )
    
    parser.add_argument(
        '--dataset',
        default='wikitext-2',
        help='Held-out text source'
    )
    
    parser.add_argument(
        '--tokens',
        type=int,
        default=30000,
        help='Total tokens to evaluate'
    )
    
    parser.add_argument(
        '--window-len',
        type=int,
        default=1024,
        help='Window length for sliding windows'
    )
    
    parser.add_argument(
        '--stride',
        type=int,
        default=512,
        help='Stride between windows'
    )
    
    parser.add_argument(
        '--drop-first-token',
        action='store_true',
        help='Drop first token in each window'
    )
    
    parser.add_argument(
        '--output-dir',
        default='results/vocab-analysis',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--cache-dir',
        default='~/models/',
        help='Directory to cache downloaded models'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Feature flags
    parser.add_argument(
        '--enable-cosine',
        action='store_true',
        help='Compute cosine variant for vocab effects'
    )
    
    parser.add_argument(
        '--enable-cascade',
        action='store_true',
        help='Enable cascade effect analysis'
    )
    
    parser.add_argument(
        '--enable-enrichment',
        action='store_true',
        help='Enable token class enrichment analysis'
    )
    
    parser.add_argument(
        '--bootstrap',
        type=int,
        default=0,
        help='Number of bootstrap samples for confidence intervals (0 to disable)'
    )
    
    # Detection parameters
    parser.add_argument(
        '--spike-threshold',
        type=float,
        default=None,
        help='Spike threshold for super weight detection. If not provided, uses model-specific defaults: '
             'Llama models (120), Phi models (150), OLMo models (70), Mistral models (100), Others (50)'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=10,
        help='Maximum iterations for super weight detection'
    )
    
    parser.add_argument(
        '--router-analysis-samples',
        type=int,
        default=5,
        help='Number of samples for MoE router analysis'
    )
    
    parser.add_argument(
        '--p-active-floor',
        type=float,
        default=0.01,
        help='Minimum expert activation probability for MoE'
    )
    
    parser.add_argument(
        '--co-spike-threshold',
        type=float,
        default=0.12,
        help='Co-spike threshold for MoE detection'
    )
    
    # Logging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress most output'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    
    args = parse_arguments()
    
    # Setup logging level
    if args.quiet:
        log_level = logging.WARNING
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    # Create analysis runner
    runner = VocabularyAnalysisRunner(
        output_dir=args.output_dir,
        seed=args.seed,
        log_level=log_level
    )
    
    # Prepare detection configuration
    detection_config = {
        'spike_threshold': args.spike_threshold,
        'max_iterations': args.max_iterations,
        'router_analysis_samples': args.router_analysis_samples,
        'p_active_floor': args.p_active_floor,
        'co_spike_threshold': args.co_spike_threshold,
        'enable_causal_scoring': True  # Always enable for MoE models
    }
    
    # Run analysis
    try:
        results = runner.run_analysis(
            model_names=args.models,
            dataset=args.dataset,
            tokens=args.tokens,
            window_len=args.window_len,
            stride=args.stride,
            drop_first_token=args.drop_first_token,
            enable_cosine=args.enable_cosine,
            enable_cascade=args.enable_cascade,
            enable_enrichment=args.enable_enrichment,
            bootstrap=args.bootstrap,
            detection_config=detection_config,
            cache_dir=args.cache_dir
        )
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {runner.output_dir}")
        
        # Print summary statistics
        cross_summary = results.get('cross_model_summary', {})
        if cross_summary:
            print(f"\nSummary:")
            print(f"- Models analyzed: {cross_summary.get('total_models', 0)}")
            agg_stats = cross_summary.get('aggregate_stats', {})
            if agg_stats:
                print(f"- Total super weights: {agg_stats.get('total_super_weights', 0)}")
                print(f"- Total neurons analyzed: {agg_stats.get('total_neurons_analyzed', 0)}")
                print(f"- Average super weights per model: {agg_stats.get('avg_super_weights_per_model', 0):.2f}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 1
    except Exception as e:
        print(f"Analysis failed: {e}")
        logging.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
