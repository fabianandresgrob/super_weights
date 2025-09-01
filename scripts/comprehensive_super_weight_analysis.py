#!/usr/bin/env python3
"""
Comprehensive Super Weight Analysis Script

This script performs automated detection and analysis of super weights across language models.
It provides functionality similar to the validation notebook but in an automated, reproducible format.

Features:
- Detect super weights with configurable parameters
- Analyze perplexity impact on WikiText
- Measure accuracy impact on MMLU, HellaSwag, and ARC
- Test individual and combined super weight effects
- Generate comprehensive reports and visualizations
- Save results in organized JSON format
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from research.researcher import SuperWeightResearchSession
from utils.datasets import DatasetLoader
from detection.super_weight import SuperWeight, MoESuperWeight

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


class SuperWeightAnalysisRunner:
    """Main class for running comprehensive super weight analysis"""
    
    def __init__(self, 
                 results_dir: str = "results/comprehensive_super_weight_analysis",
                 plots_dir: str = "results/plots",
                 log_level: int = logging.INFO):
        self.results_dir = Path(results_dir)
        self.plots_dir = Path(plots_dir)
        self.log_level = log_level
        
        # Setup logging (will be updated per model)
        self.logger = self._setup_logger()
        
        # Create base directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset loader
        self.dataset_loader = DatasetLoader()
        
    def _setup_logger(self, model_name: Optional[str] = None) -> logging.Logger:
        """Setup logging configuration with optional model-specific file handler"""
        logger = logging.getLogger('SuperWeightAnalysis')
        logger.setLevel(self.log_level)
        
        # Clear existing handlers to avoid duplication
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Add file handler for model-specific logging
        if model_name:
            clean_name = model_name.replace('/', '_').replace('-', '_')
            model_dir = self.results_dir / clean_name
            logs_dir = model_dir / 'logs'
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = logs_dir / f"analysis_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"Logging to file: {log_file}")
        
        return logger
    
    def _setup_model_directories(self, model_name: str) -> Path:
        """Setup directory structure for a specific model"""
        clean_name = model_name.replace('/', '_').replace('-', '_')
        model_dir = self.results_dir / clean_name
        
        # Create organized subdirectories
        data_dir = model_dir / 'data'
        plots_dir = model_dir / 'plots'
        logs_dir = model_dir / 'logs'
        
        for directory in [data_dir, plots_dir, logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        return model_dir
    
    def run_analysis(self, 
                    model_names: List[str],
                    detection_config: Dict[str, Any],
                    evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive analysis on multiple models
        
        Args:
            model_names: List of model names to analyze
            detection_config: Configuration for super weight detection
            evaluation_config: Configuration for evaluation tasks
            
        Returns:
            Dictionary with analysis results for all models
        """
        
        all_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_names': model_names,
                'detection_config': detection_config,
                'evaluation_config': evaluation_config
            },
            'model_results': {},
            'cross_model_analysis': {}
        }
        
        self.logger.info(f"Starting analysis for {len(model_names)} models")
        
        # Analyze each model with progress bar
        for model_name in tqdm(model_names, desc="Analyzing models"):
            self.logger.info(f"Analyzing model: {model_name}")
            
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
            
            # Setup model-specific logging early
            self._setup_model_directories(model_name)
            self.logger = self._setup_logger(model_name)
            
            try:
                model_results = self._analyze_single_model(
                    model_name, model_detection_config, evaluation_config
                )
                all_results['model_results'][model_name] = model_results
                
            except Exception as e:
                self.logger.error(f"Failed to analyze {model_name}: {e}")
                all_results['model_results'][model_name] = {
                    'error': str(e),
                    'status': 'failed',
                    'timestamp': datetime.now().isoformat()
                }
        
        # Perform cross-model analysis
        if len([r for r in all_results['model_results'].values() if 'error' not in r]) > 1:
            self.logger.info("Performing cross-model analysis")
            all_results['cross_model_analysis'] = self._cross_model_analysis(
                all_results['model_results']
            )
        
        # Generate summary plots
        self._generate_summary_plots(all_results)
        
        # Save complete results
        results_file = self.results_dir / f"complete_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        self.logger.info(f"Analysis complete. Results saved to {results_file}")
        return all_results
    
    def _analyze_single_model(self, 
                            model_name: str,
                            detection_config: Dict[str, Any],
                            evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single model"""
        
        # Log initial GPU state
        self._log_gpu_memory(f"before loading {model_name}")
        self._reset_gpu_memory_stats()
        
        # Setup model-specific directories first
        model_dir = self._setup_model_directories(model_name)
        
        model_results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'status': 'started',
            'model_directory': str(model_dir)
        }
        
        # Initialize research session
        self.logger.info(f"Loading model: {model_name}")
        try:
            session = SuperWeightResearchSession.from_model_name(
                model_name, 
                cache_dir='~/models/',
                model_kwargs={'trust_remote_code': True, 'device_map': 'auto', 'torch_dtype': torch.float16}
            )
            model_results['model_info'] = session.model_info
            
            # Log GPU usage after model loading
            self._log_gpu_memory(f"after loading {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            model_results['status'] = 'failed'
            model_results['error'] = str(e)
            return model_results
        
        try:
            # Detect super weights
            self.logger.info("Detecting super weights")
            super_weights = session.detect_super_weights(**detection_config)
            
            # Get iteration data from detector
            iteration_data = session.detector.get_iteration_data()
            
            model_results.update({
                'detection_results': {
                    'num_super_weights': len(super_weights),
                    'super_weights': [self._serialize_super_weight(sw) for sw in super_weights],
                    'detection_config': detection_config,
                    'iteration_data': iteration_data
                },
                'status': 'detection_complete'
            })
            
            if not super_weights:
                self.logger.warning(f"No super weights detected for {model_name}")
                model_results['status'] = 'no_super_weights'
                model_results['message'] = 'No super weights detected - skipping detailed analysis'
                
                # Still save the results (detection info and model info)
                self._save_model_results(model_name, model_results)
                return model_results
            
            # Analyze individual super weights
            self.logger.info("Analyzing individual super weights")
            individual_analyses = {}
            
            # Use tqdm for individual super weight analysis
            for i, sw in enumerate(tqdm(super_weights, desc="Individual super weights", leave=False)):
                self.logger.info(f"Analyzing super weight {i+1}/{len(super_weights)}: {sw}")
                individual_analyses[f"sw_{i}"] = self._analyze_individual_super_weight(
                    session, sw, evaluation_config
                )
            
            model_results['individual_analyses'] = individual_analyses
            
            # Analyze combined super weights
            if len(super_weights) > 1:
                self.logger.info("Analyzing combined super weights")
                model_results['combined_analysis'] = self._analyze_combined_super_weights(
                    session, super_weights, evaluation_config
                )
            
            # Generate model-specific plots (plots dir will be set in _save_model_results)
            # Save results first to set up directories
            self._save_model_results(model_name, model_results)
            
            # Generate plots using the model-specific plots directory
            self._generate_model_plots(model_name, model_results)
            
            model_results['status'] = 'complete'
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {model_name}: {e}")
            model_results['status'] = 'failed'
            model_results['error'] = str(e)
            
            # Save failed results too so we have logs
            try:
                self._save_model_results(model_name, model_results)
            except Exception as save_error:
                self.logger.error(f"Failed to save error results for {model_name}: {save_error}")
    
        finally:
            # Enhanced GPU cleanup
            try:
                if 'session' in locals():
                    # Clear model and tokenizer references
                    if hasattr(session, 'model'):
                        del session.model
                    if hasattr(session, 'tokenizer'):
                        del session.tokenizer
                    if hasattr(session, 'detector'):
                        del session.detector
                    if hasattr(session, 'manager'):
                        del session.manager
                    if hasattr(session, 'analyzer'):
                        del session.analyzer
                    del session
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear GPU cache multiple times for thorough cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Wait for all operations to complete
                    torch.cuda.empty_cache()  # Second cleanup after sync
                
                # Log final GPU state
                self._log_gpu_memory(f"after cleanup {model_name}")
                
            except Exception as cleanup_error:
                self.logger.warning(f"GPU cleanup warning: {cleanup_error}")
        
        return model_results
    
    def _serialize_super_weight(self, sw: SuperWeight) -> Dict[str, Any]:
        """Convert SuperWeight to serializable dictionary"""
        def safe_item(value):
            """Safely extract item from tensor or return value if already a scalar"""
            if value is None:
                return None
            if hasattr(value, 'item'):
                return value.item()
            return float(value)
        
        result = {
            'layer': sw.layer,
            'row': sw.row,
            'column': sw.column,
            'component': sw.component,
            'input_value': safe_item(sw.input_value),
            'output_value': safe_item(sw.output_value),
            'iteration_found': sw.iteration_found,
            'magnitude_product': safe_item(sw.magnitude_product),
            'coordinates': sw.coordinates,
            'original_value': safe_item(sw.original_value)
        }
        
        # Add MoE-specific fields if applicable
        if isinstance(sw, MoESuperWeight):
            result.update({
                'expert_id': sw.expert_id,
                'routing_weight': safe_item(sw.routing_weight),
                'expert_activation_rank': sw.expert_activation_rank,
                'router_confidence': safe_item(sw.router_confidence),
                'p_active': safe_item(sw.p_active),
                'routing_entropy': safe_item(sw.routing_entropy),
                'capacity_overflow_rate': safe_item(sw.capacity_overflow_rate),
                'co_spike_score': safe_item(sw.co_spike_score),
                'routed_tokens_count': sw.routed_tokens_count,
                'impact_natural': safe_item(sw.impact_natural),
                'impact_interventional': safe_item(sw.impact_interventional),
                'energy_reduction': safe_item(sw.energy_reduction),
                'stopword_skew': safe_item(sw.stopword_skew),
                'causal_agreement': safe_item(sw.causal_agreement),
                'routing_stability': safe_item(sw.routing_stability),
                'is_shared_expert': sw.is_shared_expert
            })
        
        return result
    
    def _analyze_individual_super_weight(self, 
                                       session: SuperWeightResearchSession,
                                       super_weight: SuperWeight,
                                       config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single super weight"""
        
        self.logger.info(f"Starting individual analysis for super weight: {super_weight}")
        
        analysis = {
            'super_weight': self._serialize_super_weight(super_weight),
            'timestamp': datetime.now().isoformat()
        }
        
        # Perplexity analysis
        if config.get('perplexity_analysis', True):
            self.logger.info(f"Running perplexity analysis for {super_weight}")
            try:
                perplexity_result = session.analyzer.metrics_analyzer.measure_perplexity_impact(
                    super_weight,
                    n_samples=config.get('perplexity_samples', 500)
                )
                analysis['perplexity_impact'] = perplexity_result
                self.logger.info(f"Perplexity analysis complete. Ratio: {perplexity_result.get('perplexity_ratio', 'N/A'):.3f}")
            except Exception as e:
                self.logger.warning(f"Perplexity analysis failed for {super_weight}: {e}")
                analysis['perplexity_impact'] = {'error': str(e)}
        else:
            self.logger.info("Skipping perplexity analysis (disabled)")
        
        # Accuracy analysis on multiple tasks
        accuracy_tasks = config.get('accuracy_tasks', ['hellaswag', 'arc_easy', 'mmlu'])
        analysis['accuracy_impacts'] = {}
        
        self.logger.info(f"Running accuracy analysis on {len(accuracy_tasks)} tasks: {accuracy_tasks}")
        
        # Use tqdm for task progress
        for task in tqdm(accuracy_tasks, desc="Accuracy tasks", leave=False):
            try:
                if task == 'mmlu':
                    # Test multiple MMLU subjects
                    mmlu_subjects = config.get('mmlu_subjects', ['abstract_algebra', 'anatomy', 'business_ethics'])
                    self.logger.info(f"Testing {len(mmlu_subjects)} MMLU subjects: {mmlu_subjects}")
                    mmlu_results = {}
                    
                    # Use tqdm for MMLU subjects
                    for subject in tqdm(mmlu_subjects, desc="MMLU subjects", leave=False):
                        accuracy_result = self._measure_mmlu_accuracy(
                            session, super_weight, subject, config.get('accuracy_samples', 100)
                        )
                        mmlu_results[subject] = accuracy_result
                        if 'error' not in accuracy_result:
                            self.logger.info(f"{subject}: accuracy drop {accuracy_result.get('accuracy_drop', 0):.3f}")
                        else:
                            self.logger.warning(f"{subject}: {accuracy_result['error']}")
                    analysis['accuracy_impacts']['mmlu'] = mmlu_results
                    self.logger.info(f"MMLU analysis complete")
                else:
                    accuracy_result = session.analyzer.metrics_analyzer.measure_accuracy_impact(
                        super_weight,
                        task=task,
                        n_samples=config.get('accuracy_samples', 100)
                    )
                    analysis['accuracy_impacts'][task] = accuracy_result
                    if 'error' not in accuracy_result:
                        self.logger.info(f"{task}: accuracy drop {accuracy_result.get('accuracy_drop', 0):.3f}")
                    else:
                        self.logger.warning(f"{task}: {accuracy_result['error']}")
                    
            except Exception as e:
                self.logger.warning(f"Accuracy analysis failed for {super_weight} on {task}: {e}")
                analysis['accuracy_impacts'][task] = {'error': str(e)}
        
        self.logger.info(f"Individual analysis complete for {super_weight}")
        return analysis
    
    def _measure_mmlu_accuracy(self, 
                             session: SuperWeightResearchSession,
                             super_weight: SuperWeight,
                             subject: str,
                             n_samples: int) -> Dict[str, Any]:
        """Measure MMLU accuracy for a specific subject"""
        
        # Load MMLU data
        mmlu_data = self.dataset_loader.load_mmlu(subject=subject, n_samples=n_samples)
        
        if not mmlu_data:
            return {'error': f'Could not load MMLU data for subject {subject}'}
        
        # Compute baseline accuracy
        baseline_correct = 0
        modified_correct = 0
        total = 0
        
        # Use tqdm for MMLU sample processing
        for example in tqdm(mmlu_data, desc=f"MMLU {subject}", leave=False):
            try:
                # Baseline prediction
                baseline_pred = self._predict_mmlu(session, example)
                if baseline_pred == example['label']:
                    baseline_correct += 1
                
                # Modified prediction (with super weight zeroed)
                with session.manager.temporary_zero([super_weight]):
                    modified_pred = self._predict_mmlu(session, example)
                    if modified_pred == example['label']:
                        modified_correct += 1
                
                total += 1
                
            except Exception as e:
                self.logger.debug(f"Skipped MMLU example due to error: {e}")
                continue
        
        if total == 0:
            return {'error': 'No valid examples processed'}
        
        baseline_accuracy = baseline_correct / total
        modified_accuracy = modified_correct / total
        accuracy_drop = baseline_accuracy - modified_accuracy
        
        return {
            'subject': subject,
            'baseline_accuracy': baseline_accuracy,
            'modified_accuracy': modified_accuracy,
            'accuracy_drop': accuracy_drop,
            'accuracy_ratio': modified_accuracy / baseline_accuracy if baseline_accuracy > 0 else 0.0,
            'n_samples': total,
            'impact_severity': self._classify_accuracy_impact(accuracy_drop)
        }
    
    def _predict_mmlu(self, session: SuperWeightResearchSession, example: Dict) -> int:
        """Predict MMLU answer"""
        question = example['question']
        choices = example['choices']
        
        best_score = float('-inf')
        best_idx = 0
        
        for i, choice in enumerate(choices):
            # Create question-answer format
            qa_text = f"Question: {question}\n\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\nAnswer: {choice}"
            
            # Tokenize
            tokens = session.tokenizer(qa_text, return_tensors='pt', truncation=True, max_length=512).to(session.model.device)
            
            # Compute log probability
            with torch.no_grad():
                outputs = session.model(**tokens, labels=tokens['input_ids'])
                log_prob = -outputs.loss.item()
                
                if log_prob > best_score:
                    best_score = log_prob
                    best_idx = i
        
        return best_idx
    
    def _classify_accuracy_impact(self, accuracy_drop: float) -> str:
        """Classify the severity of accuracy impact"""
        if accuracy_drop > 0.5:
            return "catastrophic"
        elif accuracy_drop > 0.2:
            return "severe"
        elif accuracy_drop > 0.1:
            return "moderate"
        elif accuracy_drop > 0.05:
            return "mild"
        else:
            return "minimal"
    
    def _analyze_combined_super_weights(self, 
                                      session: SuperWeightResearchSession,
                                      super_weights: List[SuperWeight],
                                      config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the combined effect of all super weights"""
        
        self.logger.info(f"Starting combined analysis for {len(super_weights)} super weights")
        
        analysis = {
            'num_super_weights': len(super_weights),
            'timestamp': datetime.now().isoformat()
        }
        
        # Combined perplexity analysis
        if config.get('perplexity_analysis', True):
            self.logger.info(f"Running combined perplexity analysis for all {len(super_weights)} super weights")
            try:
                perplexity_result = session.analyzer.metrics_analyzer.measure_perplexity_impact(
                    super_weights,
                    n_samples=config.get('perplexity_samples', 500)
                )
                analysis['combined_perplexity_impact'] = perplexity_result
                self.logger.info(f"Combined perplexity analysis complete. Ratio: {perplexity_result.get('perplexity_ratio', 'N/A'):.3f}")
            except Exception as e:
                self.logger.warning(f"Combined perplexity analysis failed: {e}")
                analysis['combined_perplexity_impact'] = {'error': str(e)}
        else:
            self.logger.info("Skipping combined perplexity analysis (disabled)")
        
        # Combined accuracy analysis
        accuracy_tasks = config.get('accuracy_tasks', ['hellaswag', 'arc_easy'])
        analysis['combined_accuracy_impacts'] = {}
        
        self.logger.info(f"Running combined accuracy analysis on {len(accuracy_tasks)} tasks: {accuracy_tasks}")
        
        # Use tqdm for combined accuracy tasks
        for task in tqdm(accuracy_tasks, desc="Combined accuracy tasks", leave=False):
            try:
                if task == 'mmlu':
                    # Average across multiple subjects for combined analysis
                    mmlu_subjects = config.get('mmlu_subjects', ['abstract_algebra', 'anatomy', 'business_ethics'])
                    self.logger.info(f"Testing combined effect on MMLU (averaging across subjects)")
                    combined_accuracy_result = session.analyzer.metrics_analyzer.measure_accuracy_impact(
                        super_weights,
                        task=task,
                        n_samples=config.get('accuracy_samples', 100)
                    )
                    analysis['combined_accuracy_impacts']['mmlu'] = combined_accuracy_result
                    if 'error' not in combined_accuracy_result:
                        self.logger.info(f"Combined MMLU: accuracy drop {combined_accuracy_result.get('accuracy_drop', 0):.3f}")
                    else:
                        self.logger.warning(f"Combined MMLU: {combined_accuracy_result['error']}")
                else:
                    accuracy_result = session.analyzer.metrics_analyzer.measure_accuracy_impact(
                        super_weights,
                        task=task,
                        n_samples=config.get('accuracy_samples', 100)
                    )
                    analysis['combined_accuracy_impacts'][task] = accuracy_result
                    if 'error' not in accuracy_result:
                        self.logger.info(f"{task}: accuracy drop {accuracy_result.get('accuracy_drop', 0):.3f}")
                    else:
                        self.logger.warning(f"{task}: {accuracy_result['error']}")
                    
            except Exception as e:
                self.logger.warning(f"Combined accuracy analysis failed for {task}: {e}")
                analysis['combined_accuracy_impacts'][task] = {'error': str(e)}
        
        self.logger.info(f"Combined analysis complete for {len(super_weights)} super weights")
        return analysis
    
    def _cross_model_analysis(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Perform cross-model analysis"""
        
        successful_models = {name: results for name, results in model_results.items() 
                           if 'error' not in results and results.get('status') == 'complete'}
        
        if len(successful_models) < 2:
            return {'error': 'Not enough successful models for cross-analysis'}
        
        analysis = {
            'models_analyzed': list(successful_models.keys()),
            'summary_statistics': self._compute_cross_model_statistics(successful_models)
        }
        
        return analysis
    
    def _compute_cross_model_statistics(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compute statistics across models"""
        
        stats = {
            'super_weight_counts': {},
            'perplexity_impacts': {},
            'accuracy_impacts': {},
            'architecture_distribution': {}
        }
        
        for model_name, results in model_results.items():
            # Super weight count
            num_sw = results['detection_results']['num_super_weights']
            stats['super_weight_counts'][model_name] = num_sw
            
            # Architecture info
            if 'model_info' in results:
                arch_type = results['model_info'].get('architecture_type', 'unknown')
                if arch_type not in stats['architecture_distribution']:
                    stats['architecture_distribution'][arch_type] = 0
                stats['architecture_distribution'][arch_type] += 1
            
            # Perplexity impacts
            if 'combined_analysis' in results and 'combined_perplexity_impact' in results['combined_analysis']:
                perp_result = results['combined_analysis']['combined_perplexity_impact']
                if 'perplexity_ratio' in perp_result:
                    stats['perplexity_impacts'][model_name] = perp_result['perplexity_ratio']
        
        # Summary statistics
        sw_counts = list(stats['super_weight_counts'].values())
        perp_ratios = list(stats['perplexity_impacts'].values())
        
        stats['summary'] = {
            'total_models': len(model_results),
            'avg_super_weights': np.mean(sw_counts) if sw_counts else 0,
            'std_super_weights': np.std(sw_counts) if sw_counts else 0,
            'avg_perplexity_ratio': np.mean(perp_ratios) if perp_ratios else 0,
            'std_perplexity_ratio': np.std(perp_ratios) if perp_ratios else 0
        }
        
        return stats
    
    def _generate_model_plots(self, model_name: str, results: Dict[str, Any]):
        """Generate plots for a single model using the model-specific plots directory"""
        
        # Use the plots directory set up in _save_model_results
        if not hasattr(self, '_current_model_plots_dir'):
            # Fallback if directory not set
            clean_name = model_name.replace('/', '_').replace('-', '_')
            model_dir = self.results_dir / clean_name / 'plots'
            model_dir.mkdir(parents=True, exist_ok=True)
            plots_dir = model_dir
        else:
            plots_dir = self._current_model_plots_dir
        
        # Generate timestamp for plot filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # 1. Detection iterations plot
            if 'detection_results' in results:
                iteration_data = results['detection_results'].get('iteration_data', [])
                spike_threshold = results['detection_results']['detection_config'].get('spike_threshold', 50.0)
                if iteration_data:
                    self._plot_detection_iterations(
                        iteration_data,
                        plots_dir / f'detection_iterations_{timestamp}.png',
                        spike_threshold=spike_threshold
                    )
                else:
                    self.logger.warning("No iteration data found for detection plot")
            
            # 2. Super weight distribution
            if 'detection_results' in results:
                super_weights = results['detection_results']['super_weights']
                if super_weights:
                    self._plot_super_weight_distribution(
                        super_weights,
                        plots_dir / f'super_weight_distribution_{timestamp}.png'
                    )
            
            # 3. Impact comparison
            if 'individual_analyses' in results:
                self._plot_impact_comparison(
                    results['individual_analyses'],
                    plots_dir / f'impact_comparison_{timestamp}.png'
                )
            
            # 4. Combined analysis plot
            if 'combined_analysis' in results:
                self._plot_combined_analysis(
                    results['combined_analysis'],
                    plots_dir / f'combined_analysis_{timestamp}.png'
                )
            
            # 5. MoE-specific plots if applicable
            if 'detection_results' in results and any('expert_id' in sw for sw in results['detection_results']['super_weights']):
                self._plot_moe_analysis(
                    results['detection_results']['super_weights'],
                    plots_dir / f'moe_analysis_{timestamp}.png'
                )
            
            self.logger.info(f"Plots generated for {model_name} in {plots_dir}")
        
        except Exception as e:
            self.logger.warning(f"Plot generation failed for {model_name}: {e}")

    def _plot_detection_iterations(self, iteration_data: List[Dict], save_path: Path, spike_threshold: Optional[int] = 50.0):
        """Plot super weight detection across iterations with max activations per layer"""
        
        if not iteration_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Colors for different iterations
        colors = plt.cm.viridis(np.linspace(0, 1, len(iteration_data)))
        
        # Extract data for all iterations
        layer_indices = None
        all_input_activations = []
        all_output_activations = []
        
        for iter_idx, data in enumerate(iteration_data):
            # Get layer indices (number of layers in the model)
            if layer_indices is None:
                num_layers = len(data.get('max_input_values', []))
                layer_indices = list(range(num_layers))
            
            # Extract max activations for this iteration
            input_activations = data.get('max_input_values', [])
            output_activations = data.get('max_output_values', [])
            
            if input_activations and output_activations:
                # Store for star marking
                all_input_activations.append([abs(val) for val in input_activations])
                all_output_activations.append([abs(val) for val in output_activations])
                
                # Plot input activations
                ax1.plot(layer_indices, [abs(val) for val in input_activations], 'o-', 
                        color=colors[iter_idx], alpha=0.7, linewidth=1.5,
                        label=f'Iteration {iter_idx + 1}')
                
                # Plot output activations
                ax2.plot(layer_indices, [abs(val) for val in output_activations], 'o-', 
                        color=colors[iter_idx], alpha=0.7, linewidth=1.5,
                        label=f'Iteration {iter_idx + 1}')
        
        # Mark super weight layers with stars (use maximum activation across all iterations)
        if all_input_activations and layer_indices:
            max_input_per_layer = [max(all_input_activations[i][layer] for i in range(len(all_input_activations))) 
                                 for layer in range(len(layer_indices))]
            max_output_per_layer = [max(all_output_activations[i][layer] for i in range(len(all_output_activations))) 
                                  for layer in range(len(layer_indices))]
            
            # Find layers with significant activations (potential super weight layers)
            for layer_idx in range(len(layer_indices)):
                if max_input_per_layer[layer_idx] > spike_threshold and max_output_per_layer[layer_idx] > spike_threshold:
                    ax1.scatter(layer_idx, max_input_per_layer[layer_idx], marker='*', s=100, color='red', 
                              edgecolors='black', linewidth=1, zorder=5)
                    ax2.scatter(layer_idx, max_output_per_layer[layer_idx], marker='*', s=100, color='red', 
                              edgecolors='black', linewidth=1, zorder=5)
        
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Max Absolute Input Activation')
        ax1.set_title('Maximum Input Activations per Layer')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Max Absolute Output Activation')
        ax2.set_title('Maximum Output Activations per Layer')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_super_weight_distribution(self, super_weights: List[Dict], save_path: Path):
        """Plot super weight input vs output values"""
        
        if not super_weights:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Input vs Output values
        input_values = [sw['input_value'] for sw in super_weights]
        output_values = [sw['output_value'] for sw in super_weights]
        layers = [sw['layer'] for sw in super_weights]
        
        # Color each super weight differently (using discrete colors)
        colors = plt.cm.tab10(np.arange(len(super_weights)) % 10)
        
        # Plot each super weight with a different color and label
        for i, (x, y, layer) in enumerate(zip(input_values, output_values, layers)):
            ax.scatter(x, y, c=[colors[i]], s=80, alpha=0.7, label=f'Layer {layer}', edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Input Value')
        ax.set_ylabel('Output Value')
        ax.set_title('Super Weight Input vs Output Values')
        ax.grid(True, alpha=0.3)
        
        # Add legend (but limit it if too many super weights)
        if len(super_weights) <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # For many super weights, just show a sample in legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:10], labels[:10], bbox_to_anchor=(1.05, 1), loc='upper left', 
                     title=f'Layers (showing 10/{len(super_weights)})')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_combined_analysis(self, combined_analysis: Dict[str, Any], save_path: Path):
        """Plot combined super weight removal effects"""
        
        if not combined_analysis:
            return
        
        # Collect data for plotting
        perplexity_baseline = None
        perplexity_combined = None
        accuracy_data = {}  # task -> {'baseline': value, 'combined': value}
        
        # Extract perplexity data
        if 'combined_perplexity_impact' in combined_analysis:
            perp_impact = combined_analysis['combined_perplexity_impact']
            if 'baseline_perplexity' in perp_impact and 'modified_perplexity' in perp_impact:
                perplexity_baseline = perp_impact['baseline_perplexity']
                perplexity_combined = perp_impact['modified_perplexity']
        
        # Extract accuracy data
        if 'combined_accuracy_impacts' in combined_analysis:
            for task, result in combined_analysis['combined_accuracy_impacts'].items():
                if isinstance(result, dict) and 'baseline_accuracy' in result and 'modified_accuracy' in result:
                    accuracy_data[task] = {
                        'baseline': result['baseline_accuracy'],
                        'combined': result['modified_accuracy']
                    }
                elif isinstance(result, dict) and task == 'mmlu':
                    # Handle MMLU results (average across subjects)
                    baseline_accs = []
                    modified_accs = []
                    for subject_result in result.values():
                        if isinstance(subject_result, dict) and 'baseline_accuracy' in subject_result:
                            baseline_accs.append(subject_result['baseline_accuracy'])
                            modified_accs.append(subject_result['modified_accuracy'])
                    
                    if baseline_accs:
                        accuracy_data[task] = {
                            'baseline': np.mean(baseline_accs),
                            'combined': np.mean(modified_accs)
                        }
        
        # Determine subplot layout
        num_plots = (1 if perplexity_baseline is not None else 0) + (1 if accuracy_data else 0)
        if num_plots == 0:
            return
        
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot perplexity comparison
        if perplexity_baseline is not None:
            ax = axes[plot_idx]
            values = [perplexity_baseline, perplexity_combined]
            bars = ax.bar(['Baseline', 'All Super Weights Removed'], 
                         values, color=['blue', 'orange'], alpha=0.7)
            
            # Use log scale if values vary significantly
            min_val = min(values)
            max_val = max(values)
            if max_val / min_val > 10:  # Use log scale if range is large
                ax.set_yscale('log')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel('Perplexity')
            ax.set_title('Combined Effect: Perplexity')
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot accuracy comparison (all tasks in one plot)
        if accuracy_data and plot_idx < len(axes):
            ax = axes[plot_idx]
            
            tasks = list(accuracy_data.keys())
            baseline_accs = [accuracy_data[task]['baseline'] for task in tasks]
            combined_accs = [accuracy_data[task]['combined'] for task in tasks]
            
            x_pos = range(len(tasks))
            width = 0.35
            
            ax.bar([p - width/2 for p in x_pos], baseline_accs, 
                   width, label='Baseline', alpha=0.7)
            ax.bar([p + width/2 for p in x_pos], combined_accs, 
                   width, label='All Super Weights Removed', alpha=0.7)
            
            ax.set_xlabel('Task')
            ax.set_ylabel('Accuracy')
            ax.set_title('Combined Effect: Accuracy across Tasks')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([task.upper() for task in tasks])
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_impact_comparison(self, individual_analyses: Dict[str, Dict], save_path: Path):
        """Plot comparison of baseline vs modified performance for individual super weights"""
        
        if not individual_analyses:
            return
        
        # Collect data
        perplexity_data = {'baseline': None, 'super_weights': [], 'sw_labels': []}
        accuracy_data = {}  # task -> {'baseline': value, 'super_weights': [], 'sw_labels': []}
        
        for sw_id, analysis in individual_analyses.items():
            # Perplexity data
            if 'perplexity_impact' in analysis and 'baseline_perplexity' in analysis['perplexity_impact']:
                perp_impact = analysis['perplexity_impact']
                if 'modified_perplexity' in perp_impact:
                    # Store baseline only once
                    if perplexity_data['baseline'] is None:
                        perplexity_data['baseline'] = perp_impact['baseline_perplexity']
                    perplexity_data['super_weights'].append(perp_impact['modified_perplexity'])
                    perplexity_data['sw_labels'].append(sw_id)
            
            # Accuracy data
            if 'accuracy_impacts' in analysis:
                for task, result in analysis['accuracy_impacts'].items():
                    if task not in accuracy_data:
                        accuracy_data[task] = {'baseline': None, 'super_weights': [], 'sw_labels': []}
                    
                    if isinstance(result, dict) and 'baseline_accuracy' in result and 'modified_accuracy' in result:
                        # Store baseline only once per task
                        if accuracy_data[task]['baseline'] is None:
                            accuracy_data[task]['baseline'] = result['baseline_accuracy']
                        accuracy_data[task]['super_weights'].append(result['modified_accuracy'])
                        accuracy_data[task]['sw_labels'].append(sw_id)
                    elif isinstance(result, dict) and task == 'mmlu':
                        # Handle MMLU results (average across subjects)
                        baseline_accs = []
                        modified_accs = []
                        for subject_result in result.values():
                            if isinstance(subject_result, dict) and 'baseline_accuracy' in subject_result:
                                baseline_accs.append(subject_result['baseline_accuracy'])
                                modified_accs.append(subject_result['modified_accuracy'])
                        
                        if baseline_accs:
                            # Store baseline only once per task
                            if accuracy_data[task]['baseline'] is None:
                                accuracy_data[task]['baseline'] = np.mean(baseline_accs)
                            accuracy_data[task]['super_weights'].append(np.mean(modified_accs))
                            accuracy_data[task]['sw_labels'].append(sw_id)
        
        # Determine subplot layout
        num_plots = (1 if perplexity_data['baseline'] is not None else 0) + len(accuracy_data)
        if num_plots == 0:
            return
        
        if num_plots <= 2:
            fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))
        else:
            cols = min(3, num_plots)
            rows = (num_plots + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
        
        if num_plots == 1:
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = axes.flatten()
        
        plot_idx = 0
        
        # Plot perplexity comparison
        if perplexity_data['baseline'] is not None:
            ax = axes[plot_idx]
            
            # Prepare data: baseline + all super weight results
            values = [perplexity_data['baseline']] + perplexity_data['super_weights']
            labels = ['Baseline'] + perplexity_data['sw_labels']
            colors_list = ['blue'] + ['orange'] * len(perplexity_data['super_weights'])
            
            bars = ax.bar(range(len(values)), values, color=colors_list, alpha=0.7)
            
            # Use log scale if values vary significantly
            min_val = min(values)
            max_val = max(values)
            if max_val / min_val > 10:  # Use log scale if range is large
                ax.set_yscale('log')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Perplexity')
            ax.set_title('Perplexity: Baseline vs Super Weight Removed')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot accuracy comparisons for each task
        for task, data in accuracy_data.items():
            if data['baseline'] is not None and data['super_weights'] and plot_idx < len(axes):
                ax = axes[plot_idx]
                
                # Prepare data: baseline + all super weight results
                values = [data['baseline']] + data['super_weights']
                labels = ['Baseline'] + data['sw_labels']
                colors_list = ['blue'] + ['orange'] * len(data['super_weights'])
                
                bars = ax.bar(range(len(values)), values, color=colors_list, alpha=0.7)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
                
                ax.set_xlabel('Configuration')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'{task.upper()}: Baseline vs Super Weight Removed')
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_moe_analysis(self, super_weights: List[Dict], save_path: Path):
        """Plot MoE-specific analysis"""
        
        moe_weights = [sw for sw in super_weights if sw.get('expert_id') is not None]
        
        if not moe_weights:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Expert distribution
        expert_ids = [sw['expert_id'] for sw in moe_weights]
        ax1.hist(expert_ids, bins=max(1, len(set(expert_ids))), alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Expert ID')
        ax1.set_ylabel('Number of Super Weights')
        ax1.set_title('Super Weight Distribution by Expert')
        ax1.grid(True, alpha=0.3)
        
        # P_active distribution
        p_actives = [sw['p_active'] for sw in moe_weights if sw.get('p_active') is not None]
        if p_actives:
            ax2.hist(p_actives, bins=20, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('P_active')
            ax2.set_ylabel('Count')
            ax2.set_title('Expert Activation Probability Distribution')
            ax2.grid(True, alpha=0.3)
        
        # Co-spike scores
        co_spike_scores = [sw['co_spike_score'] for sw in moe_weights if sw.get('co_spike_score') is not None]
        if co_spike_scores:
            ax3.hist(co_spike_scores, bins=20, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Co-spike Score')
            ax3.set_ylabel('Count')
            ax3.set_title('Co-spike Score Distribution')
            ax3.grid(True, alpha=0.3)
        
        # Causal agreement
        causal_agreements = [sw['causal_agreement'] for sw in moe_weights if sw.get('causal_agreement') is not None]
        if causal_agreements:
            ax4.scatter(range(len(causal_agreements)), causal_agreements, alpha=0.7)
            ax4.set_xlabel('Super Weight Index')
            ax4.set_ylabel('Causal Agreement')
            ax4.set_title('Causal Agreement Scores')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_plots(self, all_results: Dict[str, Any]):
        """Generate summary plots across all models"""
        
        successful_models = {name: results for name, results in all_results['model_results'].items() 
                           if 'error' not in results and results.get('status') == 'complete'}
        
        if len(successful_models) < 2:
            return
        
        # Create summary plots directory
        summary_plots_dir = self.results_dir / 'summary_plots'
        summary_plots_dir.mkdir(exist_ok=True)
        
        # Model comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        model_names = []
        sw_counts = []
        perp_ratios = []
        
        for model_name, results in successful_models.items():
            model_names.append(model_name.replace('/', '\n'))
            sw_counts.append(results['detection_results']['num_super_weights'])
            
            # Get combined perplexity ratio if available
            if ('combined_analysis' in results and 
                'combined_perplexity_impact' in results['combined_analysis'] and
                'perplexity_ratio' in results['combined_analysis']['combined_perplexity_impact']):
                perp_ratios.append(results['combined_analysis']['combined_perplexity_impact']['perplexity_ratio'])
            else:
                perp_ratios.append(None)
        
        # Super weight counts comparison
        ax1.bar(range(len(model_names)), sw_counts)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Number of Super Weights')
        ax1.set_title('Super Weight Count by Model')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Perplexity impact comparison
        valid_perp_ratios = [r for r in perp_ratios if r is not None]
        valid_model_names = [name for name, r in zip(model_names, perp_ratios) if r is not None]
        
        if valid_perp_ratios:
            ax2.bar(range(len(valid_perp_ratios)), valid_perp_ratios)
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Perplexity Ratio (Combined)')
            ax2.set_title('Perplexity Impact by Model')
            ax2.set_xticks(range(len(valid_model_names)))
            ax2.set_xticklabels(valid_model_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(summary_plots_dir / f'model_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Summary plots saved to {summary_plots_dir}")
    
    def _save_model_results(self, model_name: str, results: Dict[str, Any]):
        """Save results for a single model with organized directory structure"""
        
        self.logger.info(f"Starting to save results for model: {model_name}")
        
        # Clean model name for filename
        clean_name = model_name.replace('/', '_').replace('-', '_')
        model_dir = self.results_dir / clean_name
        
        # Directory structure should already be created by _setup_model_directories
        data_dir = model_dir / 'data'
        plots_dir = model_dir / 'plots'  
        logs_dir = model_dir / 'logs'
        
        # Ensure directories exist (in case called independently)
        self.logger.info(f"Ensuring directory structure exists at: {model_dir}")
        for directory in [data_dir, plots_dir, logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main results in data directory
        results_file = data_dir / f"analysis_{timestamp}.json"
        self.logger.info(f"Saving main results to: {results_file}")
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Successfully saved main results JSON ({results_file.stat().st_size} bytes)")
        except Exception as e:
            self.logger.error(f"Failed to save main results JSON: {e}")
            raise
        
        # Save individual super weight analyses
        if 'individual_analyses' in results:
            individual_count = len(results['individual_analyses'])
            self.logger.info(f"Saving {individual_count} individual super weight analyses")
            
            individual_dir = data_dir / 'individual_analyses'
            individual_dir.mkdir(exist_ok=True)
            
            # Use tqdm for individual analyses saving
            analyses_items = list(results['individual_analyses'].items())
            for sw_id, analysis in tqdm(analyses_items, desc="Saving individual analyses", leave=False):
                sw_file = individual_dir / f"{sw_id}_analysis_{timestamp}.json"
                try:
                    with open(sw_file, 'w') as f:
                        json.dump(analysis, f, indent=2, default=str)
                    self.logger.debug(f"Saved {sw_id} analysis")
                except Exception as e:
                    self.logger.warning(f"Failed to save {sw_id} analysis: {e}")
        
        # Save combined analysis if available
        if 'combined_analysis' in results:
            self.logger.info(f"Saving combined analysis")
            combined_file = data_dir / f"combined_analysis_{timestamp}.json"
            try:
                with open(combined_file, 'w') as f:
                    json.dump(results['combined_analysis'], f, indent=2, default=str)
                self.logger.info(f"Successfully saved combined analysis JSON")
            except Exception as e:
                self.logger.error(f"Failed to save combined analysis JSON: {e}")
        
        # Update plots directory reference for this model
        self._current_model_plots_dir = plots_dir
        
        self.logger.info(f"All results successfully saved for {model_name} in {model_dir}")
    
    def _get_model_specific_spike_threshold(self, model_name: str, user_threshold: Optional[float] = None) -> float:
        """Get model-specific spike threshold based on model architecture"""
        
        # If user provided a threshold, use it
        if user_threshold is not None:
            return user_threshold
        
        # Model-specific thresholds based on architecture
        model_thresholds = {
            'llama': 120.0,
            'phi': 250.0,
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
    
    def _log_gpu_memory(self, context: str = ""):
        """Log current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
            max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            
            self.logger.info(f"GPU Memory {context}: "
                            f"Allocated: {allocated:.2f}GB, "
                            f"Reserved: {reserved:.2f}GB, "
                            f"Max Allocated: {max_allocated:.2f}GB")
        else:
            self.logger.debug(f"GPU Memory {context}: CUDA not available")

    def _reset_gpu_memory_stats(self):
        """Reset GPU memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()


def parse_arguments():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Super Weight Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        required=True,
        help='Model names to analyze (space-separated)'
    )
    
    # Detection parameters
    parser.add_argument(
        '--spike-threshold', '-t',
        type=float,
        default=None,
        help='Spike threshold for super weight detection. If not provided, uses model-specific defaults: '
             'Llama models (120), Phi models (250), OLMo models (70), Mistral models (100), Others (50)'
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
        default=10,
        help='Number of samples for MoE router analysis'
    )
    
    parser.add_argument(
        '--p-active-floor',
        type=float,
        default=0.01,
        help='Minimum expert activation probability'
    )
    
    parser.add_argument(
        '--co-spike-threshold',
        type=float,
        default=0.12,
        help='Co-spike threshold for MoE detection'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--perplexity-samples',
        type=int,
        default=500,
        help='Number of samples for perplexity evaluation'
    )
    
    parser.add_argument(
        '--accuracy-samples',
        type=int,
        default=100,
        help='Number of samples for accuracy evaluation'
    )
    
    parser.add_argument(
        '--accuracy-tasks',
        nargs='+',
        default=['hellaswag', 'arc_easy', 'mmlu', 'gsm8k'],
        help='Accuracy tasks to evaluate'
    )
    
    parser.add_argument(
        '--mmlu-subjects',
        nargs='+',
        default=['abstract_algebra', 'anatomy', 'business_ethics', 'clinical_knowledge', 'college_mathematics'],
        help='MMLU subjects to test'
    )
    
    # Output configuration
    parser.add_argument(
        '--results-dir',
        default='results/comprehensive_super_weight_analysis',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--plots-dir',
        default='results/plots',
        help='Directory to save plots'
    )
    
    # Analysis options
    parser.add_argument(
        '--skip-perplexity',
        action='store_true',
        help='Skip perplexity analysis'
    )
    
    parser.add_argument(
        '--skip-accuracy',
        action='store_true',
        help='Skip accuracy analysis'
    )
    
    parser.add_argument(
        '--enable-causal-scoring',
        action='store_true',
        default=True,
        help='Enable causal scoring for MoE models'
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
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    # Create analysis runner
    runner = SuperWeightAnalysisRunner(
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        log_level=log_level
    )

    # Prepare detection configuration
    detection_config = {
        'spike_threshold': args.spike_threshold,  # Will be None if not specified by user
        'max_iterations': args.max_iterations,
        'router_analysis_samples': args.router_analysis_samples,
        'p_active_floor': args.p_active_floor,
        'co_spike_threshold': args.co_spike_threshold,
        'enable_causal_scoring': args.enable_causal_scoring
    }
    
    # Prepare evaluation configuration
    evaluation_config = {
        'perplexity_analysis': not args.skip_perplexity,
        'accuracy_analysis': not args.skip_accuracy,
        'perplexity_samples': args.perplexity_samples,
        'accuracy_samples': args.accuracy_samples,
        'accuracy_tasks': args.accuracy_tasks,
        'mmlu_subjects': args.mmlu_subjects
    }
    
    # Run analysis
    try:
        results = runner.run_analysis(
            model_names=args.models,
            detection_config=detection_config,
            evaluation_config=evaluation_config
        )
        
        # Print summary
        runner.logger.info("=== ANALYSIS SUMMARY ===")
        failed_models = []
        successful_models = []
        
        # Use tqdm for summary processing
        for model_name in tqdm(args.models, desc="Processing results", leave=False):
            if model_name in results['model_results']:
                model_result = results['model_results'][model_name]
                if 'error' not in model_result and model_result.get('status') != 'failed':
                    num_sw = model_result.get('detection_results', {}).get('num_super_weights', 0)
                    runner.logger.info(f"{model_name}: {num_sw} super weights detected [SUCCESS]")
                    successful_models.append(model_name)
                else:
                    error_msg = model_result.get('error', 'Unknown error')
                    runner.logger.error(f"{model_name}: Failed - {error_msg}")
                    failed_models.append(model_name)
            else:
                runner.logger.error(f"{model_name}: No results found")
                failed_models.append(model_name)
        
        # Return appropriate exit code
        if failed_models:
            runner.logger.error(f"Analysis failed for {len(failed_models)} model(s): {failed_models}")
            if successful_models:
                runner.logger.info(f"Analysis succeeded for {len(successful_models)} model(s): {successful_models}")
                return 1  # Partial failure
            else:
                return 1  # Complete failure
        else:
            runner.logger.info(f"Analysis succeeded for all {len(successful_models)} model(s)")
            return 0
        
    except Exception as e:
        runner.logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
