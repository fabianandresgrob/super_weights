import logging
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from detection.detector import SuperWeightDetector, MoESuperWeightDetector
from management.manager import SuperWeightManager
from analysis.analyzer import SuperWeightAnalyzer
from utils.model_architectures import UniversalMLPHandler, MLPArchitectureType


class SuperWeightResearchSession:
    """
    High-level research interface that coordinates detection, management, and analysis.
    Provides a unified entry point for super weight research workflows.
    """
    
    def __init__(self, model, tokenizer, log_level=logging.INFO):
        self.model = model
        self.tokenizer = tokenizer
        
        # Setup logging
        self.logger = self._setup_logger(log_level)
        
        # Create single MLP handler for the session
        self.mlp_handler = UniversalMLPHandler(model)
        
        # Analyze architecture once
        self.model_info = self._analyze_model_architecture()
        
        # Initialize other components with shared handler
        self.manager = SuperWeightManager(model, self.mlp_handler, log_level)  # Pass handler
        self.analyzer = SuperWeightAnalyzer(model, tokenizer, self.manager, self.mlp_handler, log_level)
        
        # Create appropriate detector based on architecture
        self.detector = self._create_detector(log_level)
        
        # State tracking
        self.detected_super_weights = []
        self.analysis_history = []
        
        self.logger.info("SuperWeightResearchSession initialized")
        self.logger.info(f"Model: {self.model_info['model_name']}")
        self.logger.info(f"Architecture: {self.model_info['architecture']}")
        self.logger.info(f"Using detector: {type(self.detector).__name__}")
    
    def _analyze_model_architecture(self) -> Dict[str, Any]:
        """Analyze model architecture using the MLP handler"""
        model_name = getattr(self.model, 'name_or_path', 'unknown').replace('/', '-')
        
        # Use MLP handler's layer-by-layer MoE detection
        moe_layers = [i for i in range(len(self.mlp_handler.layers)) if self.mlp_handler.is_moe_layer(i)]
        is_moe = len(moe_layers) > 0
        architecture = MLPArchitectureType.STANDARD_MLP.value  # Use enum value
        
        if is_moe:
            # Get architecture type from first MoE layer
            first_moe_layer_idx = moe_layers[0]
            arch_info = self.mlp_handler.get_mlp_architecture(first_moe_layer_idx)
            if arch_info.is_moe and arch_info.moe_info:
                # Use enum values directly
                architecture = arch_info.architecture_type.value
        else:
            # For non-MoE models, detect the actual architecture type
            try:
                # Get architecture from first layer
                arch_info = self.mlp_handler.get_mlp_architecture(0)
                architecture = arch_info.architecture_type.value
            except Exception:
                # Fallback to standard if detection fails
                architecture = MLPArchitectureType.STANDARD_MLP.value

        return {
            'model_name': model_name,
            'is_moe': is_moe,
            'architecture': architecture,
            'num_layers': len(self.mlp_handler.layers),
            'moe_layers': moe_layers  # Use the computed list instead of accessing _moe_layers
        }
    
    def _create_detector(self, log_level):
        """Create appropriate detector based on model architecture"""
        if self.model_info['is_moe']:
            return MoESuperWeightDetector(
                model=self.model,
                tokenizer=self.tokenizer,
                mlp_handler=self.mlp_handler,
                architecture_type=self.model_info['architecture'],
                log_level=log_level
            )
        else:
            return SuperWeightDetector(
                model=self.model,
                tokenizer=self.tokenizer,
                mlp_handler=self.mlp_handler,
                manager=self.manager,
                log_level=log_level
            )
    
    @classmethod
    def from_model_name(cls, model_name: str, 
                       model_kwargs: Dict = None,
                       tokenizer_kwargs: Dict = None,
                       cache_dir: str = None,
                       log_level=logging.INFO):
        """
        Create a research session from a model name.
        
        Args:
            model_name: HuggingFace model identifier
            model_kwargs: Optional arguments for model loading
            tokenizer_kwargs: Optional arguments for tokenizer loading
            cache_dir: Optional cache directory path (e.g., ~/models/)
            log_level: Logging level
            
        Returns:
            SuperWeightResearchSession instance
        """
        if model_kwargs is None:
            model_kwargs = {"device_map": "auto", "torch_dtype": "float16"}
        
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        # Handle cache directory and local model detection
        if cache_dir is not None:
            from pathlib import Path
            cache_path = Path(cache_dir).expanduser()

            # Check if model exists in our download format (with underscores)
            local_model_path = cache_path / model_name.replace("/", "_")
            if local_model_path.exists():
                print(f"Found locally downloaded model at {local_model_path}")
                model_name = str(local_model_path)
                model_kwargs["local_files_only"] = True
            else:
                # Use HuggingFace's cache format
                model_kwargs["cache_dir"] = str(cache_path)
                tokenizer_kwargs["cache_dir"] = str(cache_path)
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        session = cls(model, tokenizer, log_level)
        session.model_name = model_name
        
        return session
    
    def _setup_logger(self, log_level) -> logging.Logger:
        """Setup logging for the research session"""
        logger = logging.getLogger(f"SuperWeightResearch_{id(self)}")
        logger.setLevel(log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def detect_super_weights(self, 
                           input_text: str = "Apple Inc. is a worldwide tech company.",
                           spike_threshold: float = 50.0,
                           max_iterations: int = 10,
                           # Enhanced MoE parameters
                           router_analysis_samples: int = 5,
                           p_active_floor: float = 0.01,
                           co_spike_threshold: float = 0.12,
                           enable_causal_scoring: bool = True) -> List:
        """
        Detect super weights with enhanced MoE support.
        
        Args:
            input_text: Text to use for detection
            spike_threshold: Legacy threshold for dense models
            max_iterations: Maximum detection iterations
            router_analysis_samples: Number of samples for MoE router analysis
            p_active_floor: Minimum p_active threshold for expert consideration
            co_spike_threshold: Threshold for co-spike alignment score S^(l,e)(r,c)
            enable_causal_scoring: Whether to compute causal impact scores
            
        Returns:
            List of detected SuperWeight objects (or MoESuperWeight for MoE models)
        """
        self.logger.info("Starting super weight detection")
        
        # Run detection with appropriate parameters
        if self.model_info['is_moe']:
            # Enhanced MoE detection with new parameters
            self.detected_super_weights = self.detector.detect_super_weights(
                input_text=input_text,
                spike_threshold=spike_threshold,  # Keep for backward compatibility
                max_iterations=max_iterations,
                router_analysis_samples=router_analysis_samples,
                p_active_floor=p_active_floor,
                co_spike_threshold=co_spike_threshold,
                enable_causal_scoring=enable_causal_scoring
            )
        else:
            # Standard dense detection
            self.detected_super_weights = self.detector.detect_super_weights(
                input_text=input_text,
                spike_threshold=spike_threshold,
                max_iterations=max_iterations
            )
        
        self.logger.info(f"Detection complete. Found {len(self.detected_super_weights)} super weights")
        
        return self.detected_super_weights
    
    def quick_screening(self, super_weights: List = None) -> List[Dict]:
        """
        Run quick screening analysis to identify most impactful super weights.
        
        Args:
            super_weights: Optional list of super weights (uses detected if None)
            
        Returns:
            List of screening results sorted by impact
        """
        if super_weights is None:
            if not self.detected_super_weights:
                raise ValueError("No super weights available. Run detect_super_weights() first.")
            super_weights = self.detected_super_weights
        
        self.logger.info(f"Running quick screening on {len(super_weights)} super weights")
        
        screening_results = self.analyzer.screen_all_super_weights(super_weights, quick_mode=True)
        
        # Store in history
        self.analysis_history.append({
            'type': 'screening',
            'timestamp': self._get_timestamp(),
            'super_weights_count': len(super_weights),
            'results': screening_results
        })
        
        return screening_results
    
    def detailed_analysis(self, super_weight,
                         include_vocabulary: bool = True,
                         include_metrics: bool = True,
                         include_patterns: bool = False) -> Dict[str, Any]:
        """
        Run detailed analysis on a single super weight.
        
        Args:
            super_weight: SuperWeight object or index from detected list
            include_vocabulary: Whether to include vocabulary analysis
            include_metrics: Whether to include metrics analysis
            include_patterns: Whether to include pattern analysis
            
        Returns:
            Dictionary with analysis results
        """
        # Handle super_weight as index
        if isinstance(super_weight, int):
            if not self.detected_super_weights:
                raise ValueError("No super weights available. Run detect_super_weights() first.")
            super_weight = self.detected_super_weights[super_weight]
        
        self.logger.info(f"Running detailed analysis on {super_weight}")
        
        results = self.analyzer.comprehensive_analysis(
            super_weight,
            include_vocabulary=include_vocabulary,
            include_metrics=include_metrics,
            include_patterns=include_patterns
        )
        
        # Store in history
        self.analysis_history.append({
            'type': 'detailed_analysis',
            'timestamp': self._get_timestamp(),
            'super_weight': str(super_weight),
            'results': results
        })
        
        return results
    
    def batch_analysis(self, super_weights: List = None,
                      analysis_types: List[str] = None,
                      max_detailed: int = 5) -> Dict[str, Any]:
        """
        Run batch analysis on multiple super weights.
        
        Args:
            super_weights: Optional list of super weights (uses detected if None)
            analysis_types: Types of analysis to run
            max_detailed: Maximum number for detailed analysis
            
        Returns:
            Dictionary with batch analysis results
        """
        if super_weights is None:
            if not self.detected_super_weights:
                raise ValueError("No super weights available. Run detect_super_weights() first.")
            super_weights = self.detected_super_weights
        
        if analysis_types is None:
            analysis_types = ['screen', 'vocabulary', 'patterns']
        
        self.logger.info(f"Running batch analysis on {len(super_weights)} super weights")
        
        results = self.analyzer.batch_analysis(
            super_weights,
            analysis_types=analysis_types,
            max_detailed_analysis=max_detailed
        )
        
        # Store in history
        self.analysis_history.append({
            'type': 'batch_analysis',
            'timestamp': self._get_timestamp(),
            'super_weights_count': len(super_weights),
            'analysis_types': analysis_types,
            'results': results
        })
        
        return results
    
    def full_research_pipeline(self, 
                             detection_config: Dict = None,
                             screening_config: Dict = None,
                             analysis_config: Dict = None) -> Dict[str, Any]:
        """
        Run the complete research pipeline from detection to analysis.
        
        Args:
            detection_config: Configuration for detection phase
            screening_config: Configuration for screening phase  
            analysis_config: Configuration for analysis phase
            
        Returns:
            Dictionary with complete pipeline results
        """
        self.logger.info("🔬 Starting full research pipeline")
        
        # Default configurations
        if detection_config is None:
            if self.model_info['is_moe']:
                # Enhanced MoE detection config
                detection_config = {
                    'spike_threshold': 50.0,
                    'max_iterations': 10,
                    'router_analysis_samples': 8,
                    'p_active_floor': 0.01,
                    'co_spike_threshold': 0.12,
                    'enable_causal_scoring': True
                }
            else:
                # Standard dense detection config
                detection_config = {
                    'spike_threshold': 50.0,
                    'max_iterations': 10
                }
        if screening_config is None:
            screening_config = {}
        if analysis_config is None:
            analysis_config = {'analysis_types': ['screen', 'vocabulary', 'patterns']}
        
        pipeline_results = {
            'pipeline_timestamp': self._get_timestamp(),
            'model_name': getattr(self, 'model_name', 'unknown')
        }
        
        # Phase 1: Detection
        self.logger.info("🔍 Phase 1: Detection")
        detection_results = self.detect_super_weights(**detection_config)
        pipeline_results['detection'] = {
            'super_weights_found': len(detection_results),
            'super_weights': detection_results,
            'detection_config': detection_config
        }
        
        if not detection_results:
            self.logger.warning("No super weights detected. Pipeline stopping.")
            return pipeline_results
        
        # Phase 2: Quick Screening
        self.logger.info("⚡ Phase 2: Quick Screening")
        screening_results = self.quick_screening()
        pipeline_results['screening'] = screening_results[:10]  # Top 10
        
        # Phase 3: Batch Analysis
        self.logger.info("📊 Phase 3: Batch Analysis")
        batch_results = self.batch_analysis(**analysis_config)
        pipeline_results['batch_analysis'] = batch_results
        
        # Phase 4: Top Candidate Deep Dive
        self.logger.info("🎯 Phase 4: Top Candidate Analysis")
        top_candidates = screening_results[:3]  # Top 3
        detailed_results = {}
        
        for i, candidate in enumerate(top_candidates):
            sw = candidate['super_weight']
            self.logger.info(f"   Analyzing candidate {i+1}/3: {sw}")
            
            detailed_analysis = self.detailed_analysis(
                sw,
                include_vocabulary=True,
                include_metrics=True,
                include_patterns=True
            )
            detailed_results[str(sw)] = detailed_analysis
        
        pipeline_results['detailed_analysis'] = detailed_results
        
        # Phase 5: Generate Research Summary
        self.logger.info("📋 Phase 5: Research Summary")
        research_summary = self._generate_research_summary(pipeline_results)
        pipeline_results['research_summary'] = research_summary
        
        self.logger.info("✅ Full research pipeline complete")
        
        return pipeline_results
    
    def compare_with_baseline(self, test_tasks: List[str] = None) -> Dict[str, Any]:
        """
        Compare model performance with and without all detected super weights.
        
        Args:
            test_tasks: Optional list of tasks to test
            
        Returns:
            Dictionary with comparison results
        """
        if not self.detected_super_weights:
            raise ValueError("No super weights available. Run detect_super_weights() first.")
        
        if test_tasks is None:
            test_tasks = ['hellaswag']
        
        self.logger.info(f"Comparing baseline vs modified performance on {len(test_tasks)} tasks")
        
        comparison_results = {
            'total_super_weights_zeroed': len(self.detected_super_weights),
            'test_tasks': test_tasks,
            'task_results': {}
        }
        
        # Test each task
        for task in test_tasks:
            self.logger.info(f"Testing task: {task}")
            
            # Baseline performance (all super weights intact)
            baseline_result = self.analyzer.measure_accuracy_impact(
                self.detected_super_weights[0], task=task, n_samples=100
            )
            baseline_accuracy = baseline_result['baseline_accuracy']
            
            # Modified performance (all super weights zeroed)
            with self.manager.temporary_zero(self.detected_super_weights):
                # Create a dummy super weight for the analyzer (it won't be zeroed again)
                dummy_sw = self.detected_super_weights[0]
                modified_result = self.analyzer.measure_accuracy_impact(
                    dummy_sw, task=task, n_samples=100
                )
                modified_accuracy = modified_result['baseline_accuracy']  # This is actually modified
            
            comparison_results['task_results'][task] = {
                'baseline_accuracy': baseline_accuracy,
                'modified_accuracy': modified_accuracy,
                'accuracy_drop': baseline_accuracy - modified_accuracy,
                'performance_ratio': modified_accuracy / baseline_accuracy if baseline_accuracy > 0 else 0.0
            }
        
        return comparison_results
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current research session"""
        
        return {
            'model_name': getattr(self, 'model_name', 'unknown'),
            'detected_super_weights': len(self.detected_super_weights),
            'analyses_run': len(self.analysis_history),
            'current_modifications': len(self.manager.currently_modified),
            'analysis_history_summary': [
                {
                    'type': entry['type'],
                    'timestamp': entry['timestamp'],
                    'super_weights_count': entry.get('super_weights_count', 1)
                }
                for entry in self.analysis_history
            ]
        }
    
    def reset_session(self):
        """Reset the session state (restore all weights, clear history)"""
        self.logger.info("Resetting research session")
        
        # Restore all modified weights
        self.manager.restore_all()
        
        # Clear state
        self.detected_super_weights.clear()
        self.analysis_history.clear()
        
        self.logger.info("Session reset complete")
    
    def export_session(self, filename: str = None) -> str:
        """
        Export complete session data to file.
        
        Args:
            filename: Optional filename
            
        Returns:
            Path to exported file
        """
        session_data = {
            'session_summary': self.get_session_summary(),
            'detected_super_weights': [
                self.analyzer._make_serializable(sw) for sw in self.detected_super_weights
            ],
            'analysis_history': [
                self.analyzer._make_serializable(entry) for entry in self.analysis_history
            ]
        }
        
        return self.analyzer.export_results(session_data, filename)
    
    def _generate_research_summary(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a research summary from pipeline results"""
        
        summary = {
            'model_name': pipeline_results.get('model_name', 'unknown'),
            'total_super_weights': pipeline_results['detection']['super_weights_found']
        }
        
        # Screening insights
        if 'screening' in pipeline_results:
            top_result = pipeline_results['screening'][0] if pipeline_results['screening'] else {}
            summary['top_impact_score'] = top_result.get('impact_score', 0.0)
            summary['catastrophic_impact_count'] = len([
                r for r in pipeline_results['screening']
                if r.get('perplexity_analysis', {}).get('impact_severity') == 'catastrophic'
            ])
        
        # Batch analysis insights
        if 'batch_analysis' in pipeline_results:
            batch = pipeline_results['batch_analysis']
            if 'batch_summary' in batch:
                batch_summary = batch['batch_summary']
                summary.update({
                    'layers_with_super_weights': batch_summary.get('spatial_summary', {}).get('layers_with_super_weights', 0),
                    'function_type_distribution': batch_summary.get('vocabulary_summary', {}).get('function_types', {})
                })
        
        # Key findings
        summary['key_findings'] = []
        
        if summary.get('catastrophic_impact_count', 0) > 0:
            summary['key_findings'].append(f"{summary['catastrophic_impact_count']} super weights cause catastrophic impact")
        
        if summary.get('top_impact_score', 0) > 8.0:
            summary['key_findings'].append("Extremely high impact super weights detected")
        
        if summary.get('total_super_weights', 0) > 10:
            summary['key_findings'].append("Large number of super weights found")
        
        return summary
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore all weights"""
        self.manager.restore_all()


# Convenience functions for quick research workflows
def quick_super_weight_analysis(model_name: str, 
                               detection_threshold: float = 50.0,
                               max_detailed_analysis: int = 3) -> Dict[str, Any]:
    """
    Quick analysis workflow for initial super weight investigation.
    
    Args:
        model_name: Model to analyze
        detection_threshold: Threshold for super weight detection
        max_detailed_analysis: Number of super weights to analyze in detail
        
    Returns:
        Dictionary with analysis results
    """
    with SuperWeightResearchSession.from_model_name(model_name) as session:
        # Detect super weights
        super_weights = session.detect_super_weights(spike_threshold=detection_threshold)
        
        if not super_weights:
            return {'error': 'No super weights detected', 'model_name': model_name}
        
        # Quick screening
        screening_results = session.quick_screening()
        
        # Detailed analysis on top candidates
        top_candidates = screening_results[:max_detailed_analysis]
        detailed_results = {}
        
        for result in top_candidates:
            sw = result['super_weight']
            analysis = session.detailed_analysis(sw, include_patterns=True)
            detailed_results[str(sw)] = analysis
        
        return {
            'model_name': model_name,
            'total_super_weights': len(super_weights),
            'screening_results': screening_results,
            'detailed_analysis': detailed_results,
            'session_summary': session.get_session_summary()
        }


def compare_super_weights_across_models(model_names: List[str],
                                      detection_threshold: float = 50.0) -> Dict[str, Any]:
    """
    Compare super weight patterns across multiple models.
    
    Args:
        model_names: List of model names to compare
        detection_threshold: Threshold for detection
        
    Returns:
        Dictionary with comparison results
    """
    comparison_results = {
        'models_analyzed': model_names,
        'detection_threshold': detection_threshold,
        'model_results': {},
        'cross_model_analysis': {}
    }
    
    all_super_weights = {}
    
    # Analyze each model
    for model_name in model_names:
        print(f"Analyzing {model_name}...")
        
        with SuperWeightResearchSession.from_model_name(model_name) as session:
            super_weights = session.detect_super_weights(spike_threshold=detection_threshold)
            screening_results = session.quick_screening() if super_weights else []
            
            comparison_results['model_results'][model_name] = {
                'super_weights_count': len(super_weights),
                'super_weights': super_weights,
                'top_screening_results': screening_results[:5]
            }
            
            all_super_weights[model_name] = super_weights
    
    # Cross-model analysis
    comparison_results['cross_model_analysis'] = {
        'total_models': len(model_names),
        'models_with_super_weights': len([name for name, weights in all_super_weights.items() if weights]),
        'average_super_weights_per_model': sum(len(weights) for weights in all_super_weights.values()) / len(model_names),
        'coordinate_overlap': _find_coordinate_overlap(all_super_weights),
        'layer_distribution_comparison': _compare_layer_distributions(all_super_weights)
    }
    
    return comparison_results


def _find_coordinate_overlap(all_super_weights: Dict[str, List]) -> Dict[str, Any]:
    """Find overlapping super weight coordinates across models"""
    
    # Collect all coordinates by model
    model_coordinates = {}
    for model_name, super_weights in all_super_weights.items():
        coordinates = set()
        for sw in super_weights:
            coordinates.add((sw.layer, sw.row, sw.column))
        model_coordinates[model_name] = coordinates
    
    # Find universal coordinates (appear in all models)
    if model_coordinates:
        universal_coordinates = set.intersection(*model_coordinates.values()) if len(model_coordinates) > 1 else set()
        
        # Find coordinates that appear in multiple models
        all_coordinates = set()
        for coords in model_coordinates.values():
            all_coordinates.update(coords)
        
        multi_model_coordinates = {}
        for coord in all_coordinates:
            models_with_coord = [name for name, coords in model_coordinates.items() if coord in coords]
            if len(models_with_coord) > 1:
                multi_model_coordinates[coord] = models_with_coord
    else:
        universal_coordinates = set()
        multi_model_coordinates = {}
    
    return {
        'universal_coordinates': list(universal_coordinates),
        'multi_model_coordinates': {str(coord): models for coord, models in multi_model_coordinates.items()},
        'total_unique_coordinates': len(all_coordinates) if 'all_coordinates' in locals() else 0
    }


def _compare_layer_distributions(all_super_weights: Dict[str, List]) -> Dict[str, Any]:
    """Compare layer distributions across models"""
    
    layer_distributions = {}
    
    for model_name, super_weights in all_super_weights.items():
        layer_counts = {}
        for sw in super_weights:
            layer_counts[sw.layer] = layer_counts.get(sw.layer, 0) + 1
        layer_distributions[model_name] = layer_counts
    
    # Find common layers
    all_layers = set()
    for layer_counts in layer_distributions.values():
        all_layers.update(layer_counts.keys())
    
    common_layers = []
    for layer in all_layers:
        models_with_layer = [name for name, counts in layer_distributions.items() if layer in counts]
        if len(models_with_layer) > 1:
            common_layers.append({
                'layer': layer,
                'models': models_with_layer,
                'total_super_weights': sum(layer_distributions[model].get(layer, 0) for model in models_with_layer)
            })
    
    return {
        'layer_distributions': layer_distributions,
        'common_layers': sorted(common_layers, key=lambda x: x['total_super_weights'], reverse=True),
        'total_unique_layers': len(all_layers)
    }
