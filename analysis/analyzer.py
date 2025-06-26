import logging
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt

from detection.super_weight import SuperWeight
from .vocabulary import VocabularyAnalyzer
from .metrics import MetricsAnalyzer
from .patterns import PatternsAnalyzer
from .activation import SuperActivationAnalyzer


class SuperWeightAnalyzer:
    """
    Main analyzer that coordinates all analysis types for super weights.
    Provides a unified interface for vocabulary, metrics, and pattern analysis.
    """
    
    def __init__(self, model, tokenizer, manager, log_level=logging.INFO):
        self.model = model
        self.tokenizer = tokenizer
        self.manager = manager
        
        # Initialize specialized analyzers
        self.vocabulary_analyzer = VocabularyAnalyzer(model, tokenizer, manager)
        self.metrics_analyzer = MetricsAnalyzer(model, tokenizer, manager)
        self.patterns_analyzer = PatternsAnalyzer(model, tokenizer, manager)
        self.super_activation_analyzer = SuperActivationAnalyzer(model, tokenizer, log_level)
        
        # Setup logging
        self.logger = self._setup_logger(log_level)
    
    def _setup_logger(self, log_level) -> logging.Logger:
        """Setup logging for the analyzer"""
        logger = logging.getLogger(f"SuperWeightAnalyzer_{id(self)}")
        logger.setLevel(log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_vocabulary_effects(self, super_weight: SuperWeight, 
                                 test_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze vocabulary effects of a super weight.
        
        Args:
            super_weight: SuperWeight to analyze
            test_texts: Optional list of test texts
            
        Returns:
            Dictionary with vocabulary analysis results
        """
        self.logger.info(f"Analyzing vocabulary effects for {super_weight}")
        return self.vocabulary_analyzer.analyze_vocabulary_effects(super_weight, test_texts)
    
    def measure_perplexity_impact(self, super_weight: SuperWeight, 
                                dataset_name: str = 'wikitext',
                                n_samples: int = 100) -> Dict[str, Any]:
        """
        Measure perplexity impact of a super weight.
        
        Args:
            super_weight: SuperWeight to analyze
            dataset_name: Dataset to use for evaluation
            n_samples: Number of samples to evaluate
            
        Returns:
            Dictionary with perplexity analysis results
        """
        self.logger.info(f"Measuring perplexity impact for {super_weight}")
        return self.metrics_analyzer.measure_perplexity_impact(
            super_weight, dataset_name=dataset_name, n_samples=n_samples
        )
    
    def measure_accuracy_impact(self, super_weight: SuperWeight,
                              task: str = 'hellaswag',
                              n_samples: int = 100) -> Dict[str, Any]:
        """
        Measure accuracy impact on downstream tasks.
        
        Args:
            super_weight: SuperWeight to analyze
            task: Task name for evaluation
            n_samples: Number of samples to evaluate
            
        Returns:
            Dictionary with accuracy analysis results
        """
        self.logger.info(f"Measuring accuracy impact for {super_weight} on {task}")
        return self.metrics_analyzer.measure_accuracy_impact(super_weight, task, n_samples)

    def mathematical_super_activation_analysis(self, super_weight: SuperWeight, 
                                            input_text: str = "Apple Inc. is a worldwide tech company.") -> Dict[str, Any]:
        """
        Perform mathematical analysis of super activation creation.
        
        This method delegates to the SuperActivationAnalyzer which automatically detects
        the model architecture and performs the appropriate mathematical analysis.
        
        Args:
            super_weight: SuperWeight to analyze
            input_text: Input text to use for analysis
            
        Returns:
            Dictionary with mathematical analysis results
        """
        
        self.logger.info(f"Mathematical super activation analysis for {super_weight}")
        return self.super_activation_analyzer.mathematical_super_activation_analysis(super_weight, input_text)
    
    def screen_all_super_weights(self, super_weights: List[SuperWeight],
                               quick_mode: bool = True) -> List[Dict[str, Any]]:
        """
        Efficiently screen multiple super weights to identify the most impactful ones.
        
        Args:
            super_weights: List of SuperWeight objects to screen
            quick_mode: If True, use smaller sample sizes for faster screening
            
        Returns:
            List of screening results sorted by impact score
        """
        self.logger.info(f"Screening {len(super_weights)} super weights (quick_mode={quick_mode})")
        
        n_samples = 50 if quick_mode else 100
        results = []
        
        for i, sw in enumerate(super_weights):
            self.logger.info(f"Screening super weight {i+1}/{len(super_weights)}: {sw}")
            
            try:
                # Quick perplexity test
                ppl_result = self.measure_perplexity_impact(sw, n_samples=n_samples)
                
                # Quick accuracy test on HellaSwag
                acc_result = self.measure_accuracy_impact(sw, task='hellaswag', n_samples=n_samples)
                
                # Compute combined impact score
                impact_score = self._compute_screening_impact_score(ppl_result, acc_result)
                
                results.append({
                    'super_weight': sw,
                    'perplexity_analysis': ppl_result,
                    'accuracy_analysis': acc_result,
                    'impact_score': impact_score,
                    'screening_rank': 0  # Will be set after sorting
                })
                
            except Exception as e:
                self.logger.error(f"Error screening {sw}: {e}")
                results.append({
                    'super_weight': sw,
                    'error': str(e),
                    'impact_score': 0.0,
                    'screening_rank': 0
                })
        
        # Sort by impact score and assign ranks
        results.sort(key=lambda x: x.get('impact_score', 0.0), reverse=True)
        for i, result in enumerate(results):
            result['screening_rank'] = i + 1
        
        self.logger.info(f"Screening complete. Top impact score: {results[0].get('impact_score', 0.0):.3f}")
        
        return results
    
    def _compute_screening_impact_score(self, ppl_result: Dict, acc_result: Dict) -> float:
        """Compute a combined impact score for screening"""
        
        # Perplexity impact (0-10 scale)
        ppl_ratio = ppl_result.get('perplexity_ratio', 1.0)
        ppl_score = min(10.0, max(0.0, (ppl_ratio - 1.0) * 5.0))
        
        # Accuracy impact (0-10 scale)
        acc_drop = acc_result.get('accuracy_drop', 0.0)
        acc_score = min(10.0, max(0.0, acc_drop * 20.0))
        
        # Weighted combination (favor perplexity as it's more reliable)
        combined_score = 0.7 * ppl_score + 0.3 * acc_score
        
        return combined_score
    
    def comprehensive_analysis(self, super_weight: SuperWeight,
                             include_vocabulary: bool = True,
                             include_metrics: bool = True,
                             include_patterns: bool = False,
                             custom_test_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run analysis across all available methods for a single super weight.
        
        Args:
            super_weight: SuperWeight to analyze
            include_vocabulary: Whether to include vocabulary analysis
            include_metrics: Whether to include metrics analysis
            include_patterns: Whether to include pattern analysis (for single weight)
            custom_test_texts: Optional custom test texts
            
        Returns:
            Dictionary with results from all requested analyses
        """
        self.logger.info(f"Running analysis for {super_weight}")
        
        results = {
            'super_weight': super_weight,
            'analysis_timestamp': self._get_timestamp()
        }
        
        # Vocabulary analysis
        if include_vocabulary:
            self.logger.info("Running vocabulary analysis...")
            try:
                results['vocabulary_analysis'] = self.analyze_vocabulary_effects(
                    super_weight, custom_test_texts
                )
            except Exception as e:
                self.logger.error(f"Vocabulary analysis failed: {e}")
                results['vocabulary_analysis'] = {'error': str(e)}
        
        # Metrics analysis
        if include_metrics:
            self.logger.info("Running metrics analysis...")
            try:
                results['metrics_analysis'] = self.metrics_analyzer.comprehensive_impact_analysis(
                    super_weight,
                    perplexity_config={'n_samples': 100},
                    accuracy_tasks=['hellaswag']
                )
            except Exception as e:
                self.logger.error(f"Metrics analysis failed: {e}")
                results['metrics_analysis'] = {'error': str(e)}
        
        # Individual patterns analysis (activation patterns for this weight)
        if include_patterns:
            self.logger.info("Running activation pattern analysis...")
            try:
                results['activation_patterns'] = self.patterns_analyzer.analyze_activation_patterns(
                    [super_weight], custom_test_texts
                )
            except Exception as e:
                self.logger.error(f"Pattern analysis failed: {e}")
                results['activation_patterns'] = {'error': str(e)}
        
        # Generate summary
        results['analysis_summary'] = self._generate_analysis_summary(results)
        
        self.logger.info(f"Analysis complete for {super_weight}")
        
        return results
    
    def batch_analysis(self, super_weights: List[SuperWeight],
                      analysis_types: List[str] = None,
                      max_detailed_analysis: int = 5) -> Dict[str, Any]:
        """
        Run batch analysis on multiple super weights.
        
        Args:
            super_weights: List of SuperWeight objects to analyze
            analysis_types: List of analysis types ('screen', 'vocabulary', 'metrics', 'patterns')
            max_detailed_analysis: Maximum number of weights to run detailed analysis on
            
        Returns:
            Dictionary with batch analysis results
        """
        
        if analysis_types is None:
            analysis_types = ['screen', 'vocabulary', 'patterns']
        
        self.logger.info(f"Running batch analysis on {len(super_weights)} super weights")
        self.logger.info(f"Analysis types: {analysis_types}")
        
        results = {
            'total_super_weights': len(super_weights),
            'analysis_types': analysis_types,
            'analysis_timestamp': self._get_timestamp()
        }
        
        # Screening analysis
        if 'screen' in analysis_types:
            self.logger.info("Running screening analysis...")
            results['screening_results'] = self.screen_all_super_weights(super_weights)
            
            # Get top candidates for detailed analysis
            top_candidates = results['screening_results'][:max_detailed_analysis]
            top_super_weights = [r['super_weight'] for r in top_candidates]
        else:
            top_super_weights = super_weights[:max_detailed_analysis]
        
        # Vocabulary analysis on top candidates
        if 'vocabulary' in analysis_types and top_super_weights:
            self.logger.info(f"Running vocabulary analysis on top {len(top_super_weights)} candidates...")
            results['vocabulary_comparison'] = self.vocabulary_analyzer.compare_super_weights(top_super_weights)
        
        # Spatial patterns analysis
        if 'patterns' in analysis_types:
            self.logger.info("Running spatial patterns analysis...")
            results['spatial_patterns'] = self.patterns_analyzer.analyze_spatial_patterns(super_weights)
            
            # Activation patterns on top candidates
            if top_super_weights:
                results['activation_patterns'] = self.patterns_analyzer.analyze_activation_patterns(top_super_weights)
            
            # Functional patterns
            results['functional_patterns'] = self.patterns_analyzer.analyze_functional_patterns(super_weights)
        
        # Metrics analysis on top candidates
        if 'metrics' in analysis_types and top_super_weights:
            self.logger.info(f"Running metrics analysis on top {len(top_super_weights)} candidates...")
            results['metrics_results'] = []
            
            for sw in top_super_weights:
                try:
                    metrics_result = self.metrics_analyzer.comprehensive_impact_analysis(sw)
                    results['metrics_results'].append(metrics_result)
                except Exception as e:
                    self.logger.error(f"Metrics analysis failed for {sw}: {e}")
                    results['metrics_results'].append({'super_weight': sw, 'error': str(e)})
        
        # Generate batch summary
        results['batch_summary'] = self._generate_batch_summary(results)
        
        self.logger.info("Batch analysis complete")
        
        return results
    
    def _generate_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of individual super weight analysis"""
        
        summary = {
            'super_weight': str(results['super_weight']),
            'analyses_completed': []
        }
        
        # Vocabulary summary
        if 'vocabulary_analysis' in results and 'error' not in results['vocabulary_analysis']:
            vocab = results['vocabulary_analysis']
            summary['analyses_completed'].append('vocabulary')
            summary['vocabulary_summary'] = {
                'classification': vocab['classification']['type'],
                'confidence': vocab['classification']['confidence'],
                'significant_effects': vocab['statistics']['num_significant']
            }
        
        # Metrics summary
        if 'metrics_analysis' in results and 'error' not in results['metrics_analysis']:
            metrics = results['metrics_analysis']
            summary['analyses_completed'].append('metrics')
            summary['metrics_summary'] = {
                'overall_impact': metrics['overall_impact']['impact_classification'],
                'overall_score': metrics['overall_impact']['overall_score'],
                'perplexity_ratio': metrics['perplexity_analysis']['perplexity_ratio']
            }
        
        return summary
    
    def _generate_batch_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of batch analysis results"""
        
        summary = {
            'total_analyzed': results['total_super_weights'],
            'analysis_types_run': results['analysis_types']
        }
        
        # Screening summary
        if 'screening_results' in results:
            screening = results['screening_results']
            summary['screening_summary'] = {
                'top_impact_score': screening[0].get('impact_score', 0.0) if screening else 0.0,
                'high_impact_count': len([r for r in screening if r.get('impact_score', 0.0) > 5.0]),
                'catastrophic_count': len([r for r in screening 
                                        if r.get('perplexity_analysis', {}).get('impact_severity') == 'catastrophic'])
            }
        
        # Spatial patterns summary
        if 'spatial_patterns' in results:
            spatial = results['spatial_patterns']
            summary['spatial_summary'] = {
                'layers_with_super_weights': spatial['layers_with_super_weights'],
                'unique_input_channels': spatial['unique_input_channels'],
                'coordinate_clustering_score': spatial['coordinate_clusters'].get('clustering_score', 0.0)
            }
        
        # Vocabulary comparison summary
        if 'vocabulary_comparison' in results:
            vocab_comp = results['vocabulary_comparison']
            summary['vocabulary_summary'] = {
                'function_types': vocab_comp['summary']['function_type_distribution'],
                'average_similarity': vocab_comp['summary']['average_similarity']
            }
        
        return summary
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for analysis records"""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def display_analysis_results(self, analysis_results: Dict[str, Any], 
                               show_details: bool = True) -> None:
        """
        Display analysis results in a readable format.
        
        Args:
            analysis_results: Results from comprehensive_analysis or batch_analysis
            show_details: Whether to show detailed results or just summary
        """
        
        if 'total_super_weights' in analysis_results:
            # This is a batch analysis
            self._display_batch_results(analysis_results, show_details)
        else:
            # This is a single super weight analysis
            self._display_individual_results(analysis_results, show_details)
    
    def _display_individual_results(self, results: Dict[str, Any], show_details: bool) -> None:
        """Display results for individual super weight analysis"""
        
        sw = results['super_weight']
        print(f"\n{'='*60}")
        print(f"SUPER WEIGHT ANALYSIS: {sw}")
        print(f"{'='*60}")
        
        # Analysis summary
        if 'analysis_summary' in results:
            summary = results['analysis_summary']
            print(f"\nANALYSIS SUMMARY:")
            print(f"  Completed analyses: {', '.join(summary['analyses_completed'])}")
            
            if 'vocabulary_summary' in summary:
                vocab = summary['vocabulary_summary']
                print(f"  Vocabulary function: {vocab['classification']} (confidence: {vocab['confidence']:.2f})")
            
            if 'metrics_summary' in summary:
                metrics = summary['metrics_summary']
                print(f"  Impact level: {metrics['overall_impact']} (score: {metrics['overall_score']:.2f})")
                print(f"  Perplexity ratio: {metrics['perplexity_ratio']:.2f}")
        
        # Detailed results
        if show_details:
            # Vocabulary analysis details
            if 'vocabulary_analysis' in results and 'error' not in results['vocabulary_analysis']:
                print(f"\n{'-'*40}")
                print("VOCABULARY ANALYSIS DETAILS:")
                self.vocabulary_analyzer.display_analysis_results(results['vocabulary_analysis'])
            
            # Metrics analysis details
            if 'metrics_analysis' in results and 'error' not in results['metrics_analysis']:
                print(f"\n{'-'*40}")
                print("METRICS ANALYSIS DETAILS:")
                self._display_metrics_details(results['metrics_analysis'])
    
    def _display_batch_results(self, results: Dict[str, Any], show_details: bool) -> None:
        """Display results for batch analysis"""
        
        print(f"\n{'='*60}")
        print(f"BATCH SUPER WEIGHT ANALYSIS")
        print(f"{'='*60}")
        print(f"Total super weights analyzed: {results['total_super_weights']}")
        print(f"Analysis types: {', '.join(results['analysis_types'])}")
        
        # Batch summary
        if 'batch_summary' in results:
            summary = results['batch_summary']
            print(f"\nBATCH SUMMARY:")
            
            if 'screening_summary' in summary:
                screening = summary['screening_summary']
                print(f"  Top impact score: {screening['top_impact_score']:.2f}")
                print(f"  High impact super weights: {screening['high_impact_count']}")
                print(f"  Catastrophic impact super weights: {screening['catastrophic_count']}")
            
            if 'spatial_summary' in summary:
                spatial = summary['spatial_summary']
                print(f"  Layers with super weights: {spatial['layers_with_super_weights']}")
                print(f"  Unique input channels: {spatial['unique_input_channels']}")
                print(f"  Coordinate clustering score: {spatial['coordinate_clustering_score']:.3f}")
            
            if 'vocabulary_summary' in summary:
                vocab = summary['vocabulary_summary']
                print(f"  Function type distribution: {vocab['function_types']}")
                print(f"  Average similarity: {vocab['average_similarity']:.3f}")
        
        # Top screening results
        if 'screening_results' in results and show_details:
            print(f"\n{'-'*40}")
            print("TOP 10 SCREENING RESULTS:")
            
            for i, result in enumerate(results['screening_results'][:10]):
                sw = result['super_weight']
                score = result.get('impact_score', 0.0)
                ppl_ratio = result.get('perplexity_analysis', {}).get('perplexity_ratio', 1.0)
                acc_drop = result.get('accuracy_analysis', {}).get('accuracy_drop', 0.0)
                
                print(f"  {i+1:2d}. {sw}")
                print(f"      Impact score: {score:.2f}, PPL ratio: {ppl_ratio:.2f}, Acc drop: {acc_drop:.3f}")
        
        # Spatial patterns
        if 'spatial_patterns' in results and show_details:
            print(f"\n{'-'*40}")
            print("SPATIAL PATTERNS:")
            self._display_spatial_patterns(results['spatial_patterns'])
    
    def _display_metrics_details(self, metrics_results: Dict[str, Any]) -> None:
        """Display detailed metrics analysis results"""
        
        # Perplexity results
        if 'perplexity_analysis' in metrics_results:
            ppl = metrics_results['perplexity_analysis']
            print(f"  Perplexity Impact:")
            print(f"    Baseline: {ppl['baseline_perplexity']:.2f}")
            print(f"    Modified: {ppl['modified_perplexity']:.2f}")
            print(f"    Ratio: {ppl['perplexity_ratio']:.2f}")
            print(f"    Severity: {ppl['impact_severity']}")
        
        # Accuracy results
        if 'accuracy_analyses' in metrics_results:
            for task, acc in metrics_results['accuracy_analyses'].items():
                if 'error' not in acc:
                    print(f"  {task.title()} Accuracy Impact:")
                    print(f"    Baseline: {acc['baseline_accuracy']:.3f}")
                    print(f"    Modified: {acc['modified_accuracy']:.3f}")
                    print(f"    Drop: {acc['accuracy_drop']:.3f}")
                    print(f"    Severity: {acc['impact_severity']}")
        
        # Overall impact
        if 'overall_impact' in metrics_results:
            overall = metrics_results['overall_impact']
            print(f"  Overall Impact:")
            print(f"    Score: {overall['overall_score']:.2f}")
            print(f"    Classification: {overall['impact_classification']}")
    
    def _display_spatial_patterns(self, spatial_results: Dict[str, Any]) -> None:
        """Display spatial pattern analysis results"""
        
        print(f"  Layer Distribution:")
        if 'layer_distribution' in spatial_results:
            layer_dist = spatial_results['layer_distribution']
            if 'layer_counts' in layer_dist:
                for layer, count in sorted(layer_dist['layer_counts'].items()):
                    print(f"    Layer {layer}: {count} super weights")
            
            if 'most_populated_layer' in layer_dist:
                print(f"    Most populated layer: {layer_dist['most_populated_layer']}")
        
        print(f"  Channel Reuse:")
        if 'channel_reuse_patterns' in spatial_results:
            reuse = spatial_results['channel_reuse_patterns']
            print(f"    Multi-layer input channels: {reuse['multi_layer_input_channels']}")
            print(f"    Multi-layer output channels: {reuse['multi_layer_output_channels']}")
            print(f"    Shared input/output channels: {reuse['shared_input_output_channels']}")
        
        print(f"  Coordinate Clustering:")
        if 'coordinate_clusters' in spatial_results:
            clusters = spatial_results['coordinate_clusters']
            if 'error' not in clusters:
                print(f"    Average distance: {clusters['average_distance']:.2f}")
                print(f"    Clustering score: {clusters['clustering_score']:.3f}")
                print(f"    Close pairs: {clusters['close_pairs_count']}")
    
    def export_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Export analysis results to JSON file.
        
        Args:
            results: Analysis results to export
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to exported file
        """
        import json
        import os
        from datetime import datetime
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if 'total_super_weights' in results:
                filename = f"super_weight_batch_analysis_{timestamp}.json"
            else:
                sw_str = str(results['super_weight']).replace(' ', '_').replace('.', '_')
                filename = f"super_weight_analysis_{sw_str}_{timestamp}.json"
        
        # Convert any torch tensors or SuperWeight objects to serializable format
        serializable_results = self._make_serializable(results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        self.logger.info(f"Results exported to {filename}")
        return filename
    
    def _make_serializable(self, obj) -> Any:
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, SuperWeight):
            return {
                'layer': obj.layer,
                'row': obj.row,
                'column': obj.column,
                'component': obj.component,
                'input_value': obj.input_value,
                'output_value': obj.output_value,
                'iteration_found': obj.iteration_found,
                'magnitude_product': obj.magnitude_product
            }
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, 'item'):  # torch scalars
            return obj.item()
        else:
            return obj
        