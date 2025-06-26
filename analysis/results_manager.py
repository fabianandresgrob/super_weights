import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch

from detection.super_weight import SuperWeight


class VocabularyResultsManager:
    """
    Manages saving and loading of vocabulary analysis results.
    Provides structured storage for experiments with meaningful plots and JSON outputs.
    """
    
    def __init__(self, base_results_dir: str = "results"):
        self.base_results_dir = Path(base_results_dir)
        self.vocab_results_dir = self.base_results_dir / "vocabulary_analysis"
        self.plots_dir = self.vocab_results_dir / "plots"
        self.data_dir = self.vocab_results_dir / "data"
        
        # Create directories
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories for results storage"""
        self.vocab_results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def save_vocabulary_effects_analysis(self, analysis_results: Dict[str, Any], 
                                       model_name: str, 
                                       save_plots: bool = True) -> str:
        """
        Save results from analyze_vocabulary_effects.
        
        Args:
            analysis_results: Results from VocabularyAnalyzer.analyze_vocabulary_effects()
            model_name: Name of the model analyzed
            save_plots: Whether to generate and save plots
            
        Returns:
            Path to saved results file
        """
        
        super_weight = analysis_results['super_weight']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create filename
        filename = f"vocab_effects_{self._sanitize_model_name(model_name)}_{super_weight.layer}_{super_weight.row}_{super_weight.column}_{timestamp}"
        
        # Prepare data for JSON serialization
        serializable_results = self._prepare_vocab_effects_for_json(analysis_results, model_name)
        
        # Save JSON data
        json_path = self.data_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Generate plots if requested
        if save_plots:
            self._plot_vocabulary_effects(analysis_results, model_name, filename)
        
        return str(json_path)
    
    def save_neuron_vocabulary_analysis(self, analysis_results: Dict[str, Any], 
                                      model_name: str, 
                                      save_plots: bool = True) -> str:
        """
        Save results from analyze_neuron_vocabulary_effects.
        
        Args:
            analysis_results: Results from VocabularyAnalyzer.analyze_neuron_vocabulary_effects()
            model_name: Name of the model analyzed
            save_plots: Whether to generate and save plots
            
        Returns:
            Path to saved results file
        """
        
        super_weight = analysis_results['super_weight']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create filename
        filename = f"neuron_vocab_{self._sanitize_model_name(model_name)}_{super_weight.layer}_{super_weight.row}_{super_weight.column}_{timestamp}"
        
        # Prepare data for JSON serialization
        serializable_results = self._prepare_neuron_vocab_for_json(analysis_results, model_name)
        
        # Save JSON data
        json_path = self.data_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Generate plots if requested
        if save_plots:
            self._plot_neuron_vocabulary_effects(analysis_results, model_name, filename)
        
        return str(json_path)
    
    def save_cascade_analysis(self, cascade_results: Dict[str, Any], 
                            model_name: str, 
                            save_plots: bool = True) -> str:
        """
        Save results from analyze_vocabulary_cascade.
        
        Args:
            cascade_results: Results from VocabularyAnalyzer.analyze_vocabulary_cascade()
            model_name: Name of the model analyzed
            save_plots: Whether to generate and save plots
            
        Returns:
            Path to saved results file
        """
        
        super_weight = cascade_results['super_weight']
        analysis_type = cascade_results['analysis_type']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create filename
        filename = f"cascade_{analysis_type}_{self._sanitize_model_name(model_name)}_{super_weight.layer}_{super_weight.row}_{super_weight.column}_{timestamp}"
        
        # Prepare data for JSON serialization
        serializable_results = self._prepare_cascade_for_json(cascade_results, model_name)
        
        # Save JSON data
        json_path = self.data_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Generate plots if requested
        if save_plots:
            self._plot_cascade_analysis(cascade_results, model_name, filename)
        
        return str(json_path)
    
    def save_comparison_analysis(self, comparison_results: Dict[str, Any], 
                               model_name: str, 
                               comparison_type: str,
                               save_plots: bool = True) -> str:
        """
        Save comparison analysis results (neuron vs super weight, cascade methods, etc.).
        
        Args:
            comparison_results: Results from comparison methods
            model_name: Name of the model analyzed
            comparison_type: Type of comparison ('neuron_vs_super_weight', 'cascade_methods', etc.)
            save_plots: Whether to generate and save plots
            
        Returns:
            Path to saved results file
        """
        
        super_weight = comparison_results['super_weight']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create filename
        filename = f"comparison_{comparison_type}_{self._sanitize_model_name(model_name)}_{super_weight.layer}_{super_weight.row}_{super_weight.column}_{timestamp}"
        
        # Prepare data for JSON serialization
        serializable_results = self._prepare_comparison_for_json(comparison_results, model_name, comparison_type)
        
        # Save JSON data
        json_path = self.data_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Generate plots if requested
        if save_plots:
            if comparison_type == 'neuron_vs_super_weight':
                self._plot_neuron_vs_super_weight(comparison_results, model_name, filename)
            elif comparison_type == 'cascade_methods':
                self._plot_cascade_comparison(comparison_results, model_name, filename)
        
        return str(json_path)
    
    def _sanitize_model_name(self, model_name: str) -> str:
        """Sanitize model name for use in filenames"""
        return model_name.replace('/', '_').replace('-', '_')
    
    def _prepare_vocab_effects_for_json(self, analysis_results: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Prepare vocabulary effects analysis for JSON serialization"""
        
        super_weight = analysis_results['super_weight']
        
        return {
            'analysis_type': 'vocabulary_effects',
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'super_weight': {
                'layer': super_weight.layer,
                'row': super_weight.row,
                'column': super_weight.column,
                'component': super_weight.component,
                'original_value': float(super_weight.original_value) if hasattr(super_weight, 'original_value') else None
            },
            'text_source': analysis_results['text_source'],
            'statistics': analysis_results['statistics'],
            'classification': analysis_results['classification'],
            'top_tokens': {
                'top_boosted': analysis_results['top_tokens']['top_boosted'][:20],  # Limit for readability
                'top_suppressed': analysis_results['top_tokens']['top_suppressed'][:20]
            },
            'patterns': {k: v[:10] for k, v in analysis_results['patterns'].items()},  # Limit patterns
            'vocab_effects_summary': {
                'num_positive_effects': int((np.array(analysis_results['vocab_effects']) > 0).sum()),
                'num_negative_effects': int((np.array(analysis_results['vocab_effects']) < 0).sum()),
                'max_positive_effect': float(np.max(analysis_results['vocab_effects'])),
                'max_negative_effect': float(np.min(analysis_results['vocab_effects'])),
                'effect_sparsity': float((np.abs(analysis_results['vocab_effects']) > 1.0).mean())
            }
        }
    
    def _prepare_neuron_vocab_for_json(self, analysis_results: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Prepare neuron vocabulary analysis for JSON serialization"""
        
        super_weight = analysis_results['super_weight']
        
        result = {
            'analysis_type': 'neuron_vocabulary',
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'super_weight': {
                'layer': super_weight.layer,
                'row': super_weight.row,
                'column': super_weight.column,
                'component': super_weight.component,
                'original_value': float(super_weight.original_value) if hasattr(super_weight, 'original_value') else None
            }
        }
        
        if 'error' in analysis_results:
            result['error'] = analysis_results['error']
        else:
            result.update({
                'neuron_coordinates': analysis_results['neuron_coordinates'],
                'neuron_output_norm': analysis_results['neuron_output_norm'],
                'statistics': analysis_results['statistics'],
                'classification': analysis_results['classification'],
                'top_tokens': {
                    'top_boosted': analysis_results['top_tokens']['top_boosted'][:20],
                    'top_suppressed': analysis_results['top_tokens']['top_suppressed'][:20]
                },
                'patterns': {k: v[:10] for k, v in analysis_results['patterns'].items()},
                'vocab_effects_summary': {
                    'num_positive_effects': int((np.array(analysis_results['vocab_effects']) > 0).sum()),
                    'num_negative_effects': int((np.array(analysis_results['vocab_effects']) < 0).sum()),
                    'max_positive_effect': float(np.max(analysis_results['vocab_effects'])),
                    'max_negative_effect': float(np.min(analysis_results['vocab_effects'])),
                    'effect_sparsity': float((np.abs(analysis_results['vocab_effects']) > 1.0).mean())
                }
            })
        
        return result
    
    def _prepare_cascade_for_json(self, cascade_results: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Prepare cascade analysis for JSON serialization"""
        
        super_weight = cascade_results['super_weight']
        
        result = {
            'analysis_type': f"cascade_{cascade_results['analysis_type']}",
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'super_weight': {
                'layer': super_weight.layer,
                'row': super_weight.row,
                'column': super_weight.column,
                'component': super_weight.component,
                'original_value': float(super_weight.original_value) if hasattr(super_weight, 'original_value') else None
            },
            'input_text': cascade_results['input_text']
        }
        
        if 'error' in cascade_results:
            result['error'] = cascade_results['error']
        else:
            result['summary'] = cascade_results['summary']
            
            # Extract key metrics from cascade effects
            if cascade_results['analysis_type'] == 'full_projection':
                effects = cascade_results['cascade_effects']
                result['propagation_analysis'] = cascade_results['propagation_analysis']
                result['convergence_analysis'] = cascade_results['convergence_analysis']
                
                # Summarize layer effects
                result['layer_effects_summary'] = {
                    str(layer_idx): {
                        'effect_magnitude': effects[layer_idx]['effect_magnitude'],
                        'activation_magnitude': effects[layer_idx]['activation_magnitude'],
                        'layers_remaining': effects[layer_idx]['layers_remaining'],
                        'top_boosted_tokens': effects[layer_idx]['top_tokens']['top_boosted'][:5]
                    }
                    for layer_idx in sorted(effects.keys())
                }
            
            elif cascade_results['analysis_type'] == 'residual_stream':
                effects = cascade_results['residual_effects']
                result['accumulation_analysis'] = cascade_results['accumulation_analysis']
                result['amplification_analysis'] = cascade_results['amplification_analysis']
                
                # Summarize residual effects
                result['residual_effects_summary'] = {
                    str(layer_idx): {
                        'direct_effect_magnitude': effects[layer_idx]['effect_magnitude'],
                        'cumulative_magnitude': effects[layer_idx]['cumulative_magnitude'],
                        'amplification_factor': effects[layer_idx]['amplification_factor'],
                        'residual_magnitude': effects[layer_idx]['residual_magnitude'],
                        'top_boosted_tokens': effects[layer_idx]['top_tokens']['top_boosted'][:5]
                    }
                    for layer_idx in sorted(effects.keys())
                }
        
        return result
    
    def _prepare_comparison_for_json(self, comparison_results: Dict[str, Any], 
                                   model_name: str, comparison_type: str) -> Dict[str, Any]:
        """Prepare comparison analysis for JSON serialization"""
        
        super_weight = comparison_results['super_weight']
        
        result = {
            'analysis_type': f"comparison_{comparison_type}",
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'super_weight': {
                'layer': super_weight.layer,
                'row': super_weight.row,
                'column': super_weight.column,
                'component': super_weight.component,
                'original_value': float(super_weight.original_value) if hasattr(super_weight, 'original_value') else None
            }
        }
        
        if 'error' in comparison_results:
            result['error'] = comparison_results['error']
        else:
            if comparison_type == 'neuron_vs_super_weight':
                result['comparison_metrics'] = comparison_results['comparison_metrics']
                result['relationship'] = comparison_results['relationship']
                result['summary'] = comparison_results['summary']
            
            elif comparison_type == 'cascade_methods':
                result['comparison_metrics'] = comparison_results['comparison_metrics']
                result['summary'] = comparison_results['summary']
                result['input_text'] = comparison_results['input_text']
        
        return result
    
    def _plot_vocabulary_effects(self, analysis_results: Dict[str, Any], 
                               model_name: str, filename: str):
        """Generate plots for vocabulary effects analysis"""
        
        vocab_effects = analysis_results['vocab_effects']
        statistics = analysis_results['statistics']
        top_tokens = analysis_results['top_tokens']
        super_weight = analysis_results['super_weight']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Vocabulary Effects Analysis\n{model_name} - Layer {super_weight.layer} ({super_weight.row}, {super_weight.column})', fontsize=14)
        
        # 1. Distribution of vocabulary effects
        axes[0, 0].hist(vocab_effects, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Distribution of Vocabulary Effects')
        axes[0, 0].set_xlabel('Effect Magnitude')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {statistics["mean"]:.3f}\nStd: {statistics["std"]:.3f}\nKurtosis: {statistics["kurtosis"]:.2f}\nSkew: {statistics["skew"]:.2f}'
        axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Top boosted tokens
        top_boosted = top_tokens['top_boosted'][:15]
        if top_boosted:
            tokens = [f"{t['token_str'][:10]}..." if len(t['token_str']) > 10 else t['token_str'] for t in top_boosted]
            effects = [t['effect_magnitude'] for t in top_boosted]
            y_pos = range(len(tokens))
            
            axes[0, 1].barh(y_pos, effects, color='lightgreen', alpha=0.7)
            axes[0, 1].set_yticks(y_pos)
            axes[0, 1].set_yticklabels(tokens, fontsize=8)
            axes[0, 1].set_title('Top Boosted Tokens')
            axes[0, 1].set_xlabel('Effect Magnitude')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Top suppressed tokens
        top_suppressed = top_tokens['top_suppressed'][:15]
        if top_suppressed:
            tokens = [f"{t['token_str'][:10]}..." if len(t['token_str']) > 10 else t['token_str'] for t in top_suppressed]
            effects = [t['effect_magnitude'] for t in top_suppressed]
            y_pos = range(len(tokens))
            
            axes[1, 0].barh(y_pos, effects, color='lightcoral', alpha=0.7)
            axes[1, 0].set_yticks(y_pos)
            axes[1, 0].set_yticklabels(tokens, fontsize=8)
            axes[1, 0].set_title('Top Suppressed Tokens')
            axes[1, 0].set_xlabel('Effect Magnitude')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Effect magnitude vs rank (log scale)
        sorted_effects = np.sort(np.abs(vocab_effects))[::-1]
        ranks = np.arange(1, len(sorted_effects) + 1)
        
        axes[1, 1].loglog(ranks, sorted_effects, 'b-', alpha=0.7)
        axes[1, 1].set_title('Effect Magnitude vs Rank (Log-Log)')
        axes[1, 1].set_xlabel('Rank')
        axes[1, 1].set_ylabel('|Effect Magnitude|')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / f"{filename}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_neuron_vocabulary_effects(self, analysis_results: Dict[str, Any], 
                                      model_name: str, filename: str):
        """Generate plots for neuron vocabulary effects analysis"""
        
        if 'error' in analysis_results:
            return  # Skip plotting if analysis failed
        
        vocab_effects = analysis_results['vocab_effects']
        statistics = analysis_results['statistics']
        classification = analysis_results['classification']
        super_weight = analysis_results['super_weight']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Neuron Vocabulary Effects Analysis\n{model_name} - Layer {super_weight.layer} Neuron ({super_weight.row})', fontsize=14)
        
        # 1. Distribution of effects
        axes[0, 0].hist(vocab_effects, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Neuron Vocabulary Effects Distribution')
        axes[0, 0].set_xlabel('Effect Magnitude')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add classification info
        class_text = f'Type: {classification["type"]}\nConfidence: {classification["confidence"]:.2f}'
        axes[0, 0].text(0.02, 0.98, class_text, transform=axes[0, 0].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # 2. Scatter plot of positive vs negative effects
        positive_effects = vocab_effects[vocab_effects > 0]
        negative_effects = vocab_effects[vocab_effects < 0]
        
        axes[0, 1].scatter(range(len(positive_effects)), positive_effects, 
                          alpha=0.6, color='green', s=1, label='Positive')
        axes[0, 1].scatter(range(len(negative_effects)), negative_effects, 
                          alpha=0.6, color='red', s=1, label='Negative')
        axes[0, 1].set_title('Vocabulary Effects Scatter')
        axes[0, 1].set_xlabel('Token Index')
        axes[0, 1].set_ylabel('Effect Magnitude')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cumulative distribution
        sorted_effects = np.sort(vocab_effects)
        cumulative = np.arange(1, len(sorted_effects) + 1) / len(sorted_effects)
        
        axes[1, 0].plot(sorted_effects, cumulative, 'b-', alpha=0.7)
        axes[1, 0].set_title('Cumulative Distribution of Effects')
        axes[1, 0].set_xlabel('Effect Magnitude')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Effect magnitude histogram (log scale)
        abs_effects = np.abs(vocab_effects)
        abs_effects = abs_effects[abs_effects > 0]  # Remove zeros for log scale
        
        axes[1, 1].hist(abs_effects, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_title('Absolute Effect Magnitudes (Log Scale)')
        axes[1, 1].set_xlabel('|Effect Magnitude|')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / f"{filename}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cascade_analysis(self, cascade_results: Dict[str, Any], 
                             model_name: str, filename: str):
        """Generate plots for cascade analysis"""
        
        if 'error' in cascade_results:
            return  # Skip plotting if analysis failed
        
        analysis_type = cascade_results['analysis_type']
        super_weight = cascade_results['super_weight']
        
        if analysis_type == 'full_projection':
            self._plot_full_projection_cascade(cascade_results, model_name, filename)
        elif analysis_type == 'residual_stream':
            self._plot_residual_stream_cascade(cascade_results, model_name, filename)
    
    def _plot_full_projection_cascade(self, cascade_results: Dict[str, Any], 
                                    model_name: str, filename: str):
        """Plot full projection cascade analysis"""
        
        cascade_effects = cascade_results['cascade_effects']
        propagation = cascade_results['propagation_analysis']
        super_weight = cascade_results['super_weight']
        
        layer_indices = sorted(cascade_effects.keys())
        effect_magnitudes = [cascade_effects[i]['effect_magnitude'] for i in layer_indices]
        activation_magnitudes = [cascade_effects[i]['activation_magnitude'] for i in layer_indices]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Full Projection Cascade Analysis\n{model_name} - {super_weight}', fontsize=14)
        
        # 1. Effect magnitude trajectory
        axes[0, 0].plot(layer_indices, effect_magnitudes, 'b-o', alpha=0.7)
        axes[0, 0].axvline(propagation['peak_layer'], color='red', linestyle='--', alpha=0.7, label='Peak Layer')
        axes[0, 0].set_title('Effect Magnitude Through Layers')
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Effect Magnitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Activation magnitude trajectory
        axes[0, 1].plot(layer_indices, activation_magnitudes, 'g-o', alpha=0.7)
        axes[0, 1].set_title('Activation Magnitude Through Layers')
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_ylabel('Activation Magnitude')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Convergence analysis
        if 'convergence_analysis' in cascade_results:
            convergence = cascade_results['convergence_analysis']
            conv_scores = [convergence['convergence_scores'][i] for i in layer_indices]
            
            axes[1, 0].plot(layer_indices, conv_scores, 'r-o', alpha=0.7)
            axes[1, 0].axhline(0.9, color='black', linestyle='--', alpha=0.5, label='90% Similarity')
            axes[1, 0].set_title('Convergence to Final Effect')
            axes[1, 0].set_xlabel('Layer Index')
            axes[1, 0].set_ylabel('Similarity to Final')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Effect vs activation relationship
        axes[1, 1].scatter(activation_magnitudes, effect_magnitudes, alpha=0.7, color='purple')
        axes[1, 1].set_title('Effect vs Activation Magnitude')
        axes[1, 1].set_xlabel('Activation Magnitude')
        axes[1, 1].set_ylabel('Effect Magnitude')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / f"{filename}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_residual_stream_cascade(self, cascade_results: Dict[str, Any], 
                                    model_name: str, filename: str):
        """Plot residual stream cascade analysis"""
        
        residual_effects = cascade_results['residual_effects']
        accumulation = cascade_results['accumulation_analysis']
        amplification = cascade_results['amplification_analysis']
        super_weight = cascade_results['super_weight']
        
        layer_indices = sorted(residual_effects.keys())
        direct_magnitudes = [residual_effects[i]['effect_magnitude'] for i in layer_indices]
        cumulative_magnitudes = [residual_effects[i]['cumulative_magnitude'] for i in layer_indices]
        amplification_factors = [residual_effects[i]['amplification_factor'] for i in layer_indices]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Residual Stream Cascade Analysis\n{model_name} - {super_weight}', fontsize=14)
        
        # 1. Direct vs cumulative effects
        axes[0, 0].plot(layer_indices, direct_magnitudes, 'b-o', alpha=0.7, label='Direct Effects')
        axes[0, 0].plot(layer_indices, cumulative_magnitudes, 'r-o', alpha=0.7, label='Cumulative Effects')
        axes[0, 0].set_title('Direct vs Cumulative Effect Magnitudes')
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Effect Magnitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Amplification factors
        axes[0, 1].plot(layer_indices, amplification_factors, 'g-o', alpha=0.7)
        axes[0, 1].axhline(1.0, color='black', linestyle='--', alpha=0.5, label='No Amplification')
        axes[0, 1].set_title('Amplification Factors')
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_ylabel('Amplification Factor')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Contribution by layer (bar chart)
        axes[1, 0].bar(layer_indices, direct_magnitudes, alpha=0.7, color='skyblue')
        significant_layers = accumulation['significant_layers']
        for layer in significant_layers:
            if layer in layer_indices:
                idx = layer_indices.index(layer)
                axes[1, 0].bar(layer, direct_magnitudes[idx], alpha=0.9, color='orange')
        
        axes[1, 0].set_title('Direct Contributions by Layer')
        axes[1, 0].set_xlabel('Layer Index')
        axes[1, 0].set_ylabel('Direct Effect Magnitude')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Cumulative growth
        if len(cumulative_magnitudes) > 1:
            growth_rates = [cumulative_magnitudes[i] / cumulative_magnitudes[i-1] 
                           if cumulative_magnitudes[i-1] > 0 else 1.0 
                           for i in range(1, len(cumulative_magnitudes))]
            
            axes[1, 1].plot(layer_indices[1:], growth_rates, 'purple', marker='o', alpha=0.7)
            axes[1, 1].axhline(1.0, color='black', linestyle='--', alpha=0.5, label='No Growth')
            axes[1, 1].set_title('Cumulative Growth Rate')
            axes[1, 1].set_xlabel('Layer Index')
            axes[1, 1].set_ylabel('Growth Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / f"{filename}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_neuron_vs_super_weight(self, comparison_results: Dict[str, Any], 
                                   model_name: str, filename: str):
        """Plot neuron vs super weight comparison"""
        
        if 'error' in comparison_results:
            return
        
        neuron_analysis = comparison_results['neuron_analysis']
        sw_analysis = comparison_results['super_weight_analysis']
        metrics = comparison_results['comparison_metrics']
        super_weight = comparison_results['super_weight']
        
        neuron_effects = neuron_analysis['vocab_effects']
        sw_effects = sw_analysis['vocab_effects']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Neuron vs Super Weight Comparison\n{model_name} - {super_weight}', fontsize=14)
        
        # 1. Effect distributions
        axes[0, 0].hist(neuron_effects, bins=50, alpha=0.6, color='blue', label='Neuron', density=True)
        axes[0, 0].hist(sw_effects, bins=50, alpha=0.6, color='red', label='Super Weight', density=True)
        axes[0, 0].set_title('Effect Distributions')
        axes[0, 0].set_xlabel('Effect Magnitude')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Scatter plot
        axes[0, 1].scatter(neuron_effects, sw_effects, alpha=0.6, s=1)
        axes[0, 1].plot([min(neuron_effects), max(neuron_effects)], 
                       [min(neuron_effects), max(neuron_effects)], 'r--', alpha=0.7)
        axes[0, 1].set_title(f'Effect Correlation (r={metrics["correlation"]:.3f})')
        axes[0, 1].set_xlabel('Neuron Effects')
        axes[0, 1].set_ylabel('Super Weight Effects')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Top tokens comparison
        neuron_top = set(t['token_str'] for t in neuron_analysis['top_tokens']['top_boosted'][:20])
        sw_top = set(t['token_str'] for t in sw_analysis['top_tokens']['top_boosted'][:20])
        
        overlap = len(neuron_top.intersection(sw_top))
        total = len(neuron_top.union(sw_top))
        
        axes[1, 0].bar(['Overlap', 'Neuron Only', 'Super Weight Only'], 
                      [overlap, len(neuron_top) - overlap, len(sw_top) - overlap],
                      color=['green', 'blue', 'red'], alpha=0.7)
        axes[1, 0].set_title('Top Tokens Overlap')
        axes[1, 0].set_ylabel('Number of Tokens')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Metrics summary
        metrics_text = f"""Correlation: {metrics['correlation']:.3f}
Cosine Similarity: {metrics['cosine_similarity']:.3f}
Magnitude Ratio: {metrics['magnitude_ratio']:.3f}
Token Overlap: {metrics['token_overlap_ratio']:.1%}
Classification Match: {metrics['classification_match']}"""
        
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_title('Comparison Metrics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plot_path = self.plots_dir / f"{filename}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cascade_comparison(self, comparison_results: Dict[str, Any], 
                               model_name: str, filename: str):
        """Plot cascade methods comparison"""
        
        if 'comparison_error' in comparison_results:
            return
        
        fp_results = comparison_results['full_projection']
        rs_results = comparison_results['residual_stream']
        metrics = comparison_results['comparison_metrics']
        super_weight = comparison_results['super_weight']
        
        # Extract trajectories
        fp_layers = sorted(fp_results['cascade_effects'].keys())
        fp_magnitudes = [fp_results['cascade_effects'][i]['effect_magnitude'] for i in fp_layers]
        
        rs_layers = sorted(rs_results['residual_effects'].keys())
        rs_magnitudes = [rs_results['residual_effects'][i]['cumulative_magnitude'] for i in rs_layers]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Cascade Methods Comparison\n{model_name} - {super_weight}', fontsize=14)
        
        # 1. Magnitude trajectories
        axes[0, 0].plot(fp_layers, fp_magnitudes, 'b-o', alpha=0.7, label='Full Projection')
        axes[0, 1].plot(rs_layers, rs_magnitudes, 'r-o', alpha=0.7, label='Residual Stream')
        axes[0, 0].set_title('Full Projection Trajectory')
        axes[0, 1].set_title('Residual Stream Trajectory')
        for ax in axes[0, :]:
            ax.set_xlabel('Layer Index')
            ax.set_ylabel('Effect Magnitude')
            ax.grid(True, alpha=0.3)
        
        # 2. Direct comparison (aligned lengths)
        min_len = min(len(fp_magnitudes), len(rs_magnitudes))
        fp_aligned = fp_magnitudes[:min_len]
        rs_aligned = rs_magnitudes[:min_len]
        layers_aligned = range(min_len)
        
        axes[1, 0].plot(layers_aligned, fp_aligned, 'b-o', alpha=0.7, label='Full Projection')
        axes[1, 0].plot(layers_aligned, rs_aligned, 'r-o', alpha=0.7, label='Residual Stream')
        axes[1, 0].set_title('Method Comparison (Aligned)')
        axes[1, 0].set_xlabel('Layer Index')
        axes[1, 0].set_ylabel('Effect Magnitude')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 3. Comparison metrics
        metrics_text = f"""Final Correlation: {metrics['final_effects_correlation']:.3f}
Final Cosine Sim: {metrics['final_effects_cosine_similarity']:.3f}
Trajectory Correlation: {metrics['magnitude_trajectory_correlation']:.3f}
FP Final Magnitude: {metrics['fp_final_magnitude']:.4f}
RS Final Magnitude: {metrics['rs_final_magnitude']:.4f}"""
        
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                       fontsize=11, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_title('Comparison Metrics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plot_path = self.plots_dir / f"{filename}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def load_analysis_results(self, results_path: str) -> Dict[str, Any]:
        """Load previously saved analysis results"""
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def list_saved_analyses(self, analysis_type: Optional[str] = None) -> List[str]:
        """List all saved analysis files, optionally filtered by type"""
        
        pattern = "*.json"
        if analysis_type:
            pattern = f"{analysis_type}*.json"
        
        files = list(self.data_dir.glob(pattern))
        return [str(f) for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)]
    
    def create_summary_report(self, model_name: str) -> str:
        """Create a summary report of all analyses for a given model"""
        
        model_files = [f for f in self.list_saved_analyses() if self._sanitize_model_name(model_name) in f]
        
        if not model_files:
            return f"No analyses found for model: {model_name}"
        
        summary = {
            'model_name': model_name,
            'total_analyses': len(model_files),
            'analysis_types': {},
            'super_weights_analyzed': set(),
            'latest_analysis': None
        }
        
        for file_path in model_files:
            try:
                results = self.load_analysis_results(file_path)
                analysis_type = results.get('analysis_type', 'unknown')
                
                # Count analysis types
                summary['analysis_types'][analysis_type] = summary['analysis_types'].get(analysis_type, 0) + 1
                
                # Track super weights
                sw_info = results.get('super_weight', {})
                if sw_info:
                    sw_key = f"L{sw_info.get('layer', '?')}_{sw_info.get('row', '?')}_{sw_info.get('column', '?')}"
                    summary['super_weights_analyzed'].add(sw_key)
                
                # Track latest
                if not summary['latest_analysis'] or results.get('timestamp', '') > summary['latest_analysis'].get('timestamp', ''):
                    summary['latest_analysis'] = results
                    
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        summary['super_weights_analyzed'] = list(summary['super_weights_analyzed'])
        
        # Generate summary report
        report = f"""
# Vocabulary Analysis Summary Report
## Model: {model_name}

### Overview
- Total analyses: {summary['total_analyses']}
- Unique super weights analyzed: {len(summary['super_weights_analyzed'])}
- Latest analysis: {summary['latest_analysis']['timestamp'] if summary['latest_analysis'] else 'N/A'}

### Analysis Types
"""
        for analysis_type, count in summary['analysis_types'].items():
            report += f"- {analysis_type}: {count}\n"
        
        report += f"\n### Super Weights Analyzed\n"
        for sw in summary['super_weights_analyzed']:
            report += f"- {sw}\n"
        
        # Save summary report
        report_path = self.vocab_results_dir / f"summary_report_{self._sanitize_model_name(model_name)}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        return str(report_path)