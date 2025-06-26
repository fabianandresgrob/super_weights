#!/usr/bin/env python3
"""
Comprehensive Analysis Script for Super Weights Vocabulary Analysis

This script runs the complete pipeline using the researcher session approach:
1. Super weight detection on multiple models using SuperWeightResearchSession
2. Comprehensive vocabulary analysis on all detected super weights  
3. Results saving with plots for supervisor presentation

Usage:
    python run_comprehensive_analysis.py
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import torch

from research.researcher import SuperWeightResearchSession
from analysis.vocabulary import VocabularyAnalyzer


class ComprehensiveAnalysisRunner:
    """
    Runs comprehensive analysis pipeline using the researcher session approach.
    """
    
    def __init__(self, results_dir: str = "../results"):
        self.results_dir = Path(results_dir)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Setup logging
        self.setup_logging()
        
        # Create results directories
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "super_weights").mkdir(exist_ok=True)
        
    def setup_logging(self):
        """Setup logging for the analysis run"""
        log_file = self.results_dir / f"analysis_run_{self.timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Starting comprehensive analysis run: {self.timestamp}")
    
    def run_model_analysis(self, model_name: str, detection_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run complete analysis pipeline for a single model using researcher session.
        
        Args:
            model_name: Name or path of the model to analyze
            detection_params: Parameters for super weight detection
            
        Returns:
            Dictionary with analysis results and saved file paths
        """
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🔬 ANALYZING MODEL: {model_name}")
        self.logger.info(f"{'='*80}")
        
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'detection_params': detection_params or {},
            'super_weights': [],
            'vocabulary_analyses': {},
            'errors': []
        }
        
        session = None
        
        try:
            # 1. Initialize researcher session (this loads model and tokenizer)
            self.logger.info("📥 Loading model using researcher session...")
            session = SuperWeightResearchSession.from_model_name(model_name)
            self.logger.info(f"✅ Model loaded on device: {session.model.device}")
            
            # 2. Run super weight detection
            self.logger.info("🔍 Running super weight detection...")
            detection_params = detection_params or {
                'input_text': "Apple Inc. is a worldwide tech company.",
                'spike_threshold': 50.0,
                'max_iterations': 10
            }
            
            super_weights = session.detect_super_weights(**detection_params)
            self.logger.info(f"🎯 Detected {len(super_weights)} super weights")
            
            # Save detection results
            detection_results = {
                'model_name': model_name,
                'model_architecture': type(session.model).__name__,
                'num_layers': len(session.detector.layers),
                'total_super_weights': len(super_weights),
                'detection_params': detection_params,
                'super_weights': [
                    {
                        'layer': sw.layer,
                        'coordinates': [sw.row, sw.column],
                        'component': sw.component,
                        'input_value': float(sw.input_value) if hasattr(sw, 'input_value') else None,
                        'output_value': float(sw.output_value) if hasattr(sw, 'output_value') else None,
                        'iteration_found': getattr(sw, 'iteration_found', None),
                        'magnitude_product': float(sw.input_value * sw.output_value) if hasattr(sw, 'input_value') and hasattr(sw, 'output_value') else None,
                        'original_value': float(sw.original_value) if hasattr(sw, 'original_value') else None
                    }
                    for sw in super_weights
                ]
            }
            
            # Save detection results
            sanitized_name = model_name.replace('/', '_').replace('-', '_')
            detection_file = self.results_dir / "super_weights" / f"comprehensive_analysis_{sanitized_name}_{self.timestamp}.json"
            with open(detection_file, 'w') as f:
                json.dump(detection_results, f, indent=2)
            self.logger.info(f"💾 Detection results saved: {detection_file}")
            
            results['super_weights'] = super_weights
            results['detection_file'] = str(detection_file)
            
            if not super_weights:
                self.logger.warning("⚠️  No super weights detected, skipping vocabulary analysis")
                return results
            
            # 3. Run vocabulary analysis on each super weight using the new saving functionality
            self.logger.info("📚 Starting vocabulary analysis with results saving...")
            
            # Initialize vocabulary analyzer with results saving
            vocab_analyzer = VocabularyAnalyzer(session.model, session.tokenizer, session.manager, self.results_dir)
            
            vocabulary_results = {}
            
            for i, super_weight in enumerate(super_weights):
                self.logger.info(f"\n--- Analyzing super weight {i+1}/{len(super_weights)}: {super_weight} ---")
                
                try:
                    # Run complete vocabulary analysis for this super weight using the new methods
                    saved_files = vocab_analyzer.run_complete_vocabulary_analysis(
                        super_weight=super_weight,
                        model_name=model_name,
                        save_plots=True,
                        display_results=False  # Keep quiet for batch processing
                    )
                    
                    vocabulary_results[str(super_weight)] = saved_files
                    
                    # Count successful analyses
                    success_count = sum(1 for path in saved_files.values() if path is not None)
                    self.logger.info(f"✅ Completed {success_count}/{len(saved_files)} analyses for {super_weight}")
                    
                except Exception as e:
                    error_msg = f"❌ Vocabulary analysis failed for {super_weight}: {str(e)}"
                    self.logger.error(error_msg)
                    results['errors'].append(error_msg)
                    vocabulary_results[str(super_weight)] = None
            
            results['vocabulary_analyses'] = vocabulary_results
            
            # 4. Generate model summary report
            try:
                summary_report = vocab_analyzer.create_model_summary_report(model_name)
                results['summary_report'] = summary_report
                self.logger.info(f"📋 Summary report created: {summary_report}")
            except Exception as e:
                error_msg = f"❌ Summary report creation failed: {str(e)}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
            
            self.logger.info(f"✅ Model analysis completed: {model_name}")
            
        except Exception as e:
            error_msg = f"❌ Model analysis failed for {model_name}: {str(e)}"
            self.logger.error(error_msg)
            results['errors'].append(error_msg)
        
        finally:
            # Clean up model from memory
            if session:
                del session.model, session.tokenizer, session.detector, session.manager, session.analyzer
                del session
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return results
    
    def run_batch_analysis(self, model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run analysis on multiple models.
        
        Args:
            model_configs: List of dictionaries with model configuration
                          Each dict should have 'model_name' and optionally 'detection_params'
            
        Returns:
            Dictionary with results for all models
        """
        
        self.logger.info(f"🚀 Starting batch analysis on {len(model_configs)} models")
        
        batch_results = {
            'timestamp': self.timestamp,
            'total_models': len(model_configs),
            'model_results': {},
            'summary': {
                'successful_models': 0,
                'failed_models': 0,
                'total_super_weights': 0,
                'total_vocabulary_analyses': 0
            }
        }
        
        for i, config in enumerate(model_configs):
            model_name = config['model_name']
            detection_params = config.get('detection_params', None)
            
            self.logger.info(f"\n🔄 Processing model {i+1}/{len(model_configs)}: {model_name}")
            
            model_results = self.run_model_analysis(model_name, detection_params)
            batch_results['model_results'][model_name] = model_results
            
            # Update summary
            if not model_results.get('errors'):
                batch_results['summary']['successful_models'] += 1
            else:
                batch_results['summary']['failed_models'] += 1
            
            batch_results['summary']['total_super_weights'] += len(model_results.get('super_weights', []))
            
            # Count vocabulary analyses
            vocab_analyses = model_results.get('vocabulary_analyses', {})
            for sw_analyses in vocab_analyses.values():
                if sw_analyses:
                    batch_results['summary']['total_vocabulary_analyses'] += sum(
                        1 for path in sw_analyses.values() if path is not None
                    )
        
        # Save batch results summary
        batch_file = self.results_dir / f"batch_analysis_summary_{self.timestamp}.json"
        with open(batch_file, 'w') as f:
            # Convert SuperWeight objects to strings for JSON serialization
            serializable_results = self._make_json_serializable(batch_results)
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"\n📊 BATCH ANALYSIS COMPLETE")
        self.logger.info(f"✅ Successful models: {batch_results['summary']['successful_models']}")
        self.logger.info(f"❌ Failed models: {batch_results['summary']['failed_models']}")
        self.logger.info(f"🎯 Total super weights: {batch_results['summary']['total_super_weights']}")
        self.logger.info(f"📚 Total vocabulary analyses: {batch_results['summary']['total_vocabulary_analyses']}")
        self.logger.info(f"💾 Batch summary saved: {batch_file}")
        
        return batch_results
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj


def main():
    """Main function to run comprehensive analysis"""
    
    # Define models to analyze for supervisor meeting
    model_configs = [
        {
            'model_name': 'allenai/OLMo-1B-0724-hf',
            'detection_params': {
                'input_text': "Apple Inc. is a worldwide tech company.",
                'spike_threshold': 70.0,
                'max_iterations': 8
            }
        },
        # {
        #     'model_name': 'microsoft/Phi-3-mini-4k-instruct',
        #     'detection_params': {
        #         'input_text': "Apple Inc. is a worldwide tech company.",
        #         'spike_threshold': 50.0,
        #         'max_iterations': 8
        #     }
        # },
        # {
        #     'model_name': 'meta-llama/Llama-3.1-8B',
        #     'detection_params': {
        #         'input_text': "Apple Inc. is a worldwide tech company.",
        #         'spike_threshold': 50.0,
        #         'max_iterations': 8
        #     }
        # },
        # {
        #     'model_name': 'mistralai/Mistral-7B-v0.1',
        #     'detection_params': {
        #         'input_text': "Apple Inc. is a worldwide tech company.",
        #         'spike_threshold': 50.0,
        #         'max_iterations': 8
        #     }
        # },
        # {
        #     'model_name': 'allenai/OLMo-7B-0724-hf',
        #     'detection_params': {
        #         'input_text': "Apple Inc. is a worldwide tech company.",
        #         'spike_threshold': 50.0,
        #         'max_iterations': 8
        #     }
        # }
    ]
    
    print("🔬 Super Weights Comprehensive Analysis")
    print("=" * 50)
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Models to analyze: {len(model_configs)}")
    for i, config in enumerate(model_configs):
        print(f"   {i+1}. {config['model_name']}")
    print()
    
    # Initialize runner
    runner = ComprehensiveAnalysisRunner()
    
    # Run batch analysis
    try:
        results = runner.run_batch_analysis(model_configs)
        
        print("\n🎉 ANALYSIS COMPLETE!")
        print("📁 Results saved in: results/")
        print("   ├── super_weights/           # Detection results (JSON)")
        print("   ├── vocabulary_analysis/")
        print("   │   ├── data/               # Vocabulary analysis results (JSON)")
        print("   │   ├── plots/              # Analysis plots (PNG)")
        print("   │   └── summary_report_*.md # Model summary reports")
        print("   └── batch_analysis_summary_*.json")
        print()
        print("🎯 Ready for supervisor meeting!")
        print()
        print("📊 Quick Summary:")
        summary = results['summary']
        print(f"   ✅ Successful models: {summary['successful_models']}/{summary['successful_models'] + summary['failed_models']}")
        print(f"   🎯 Total super weights found: {summary['total_super_weights']}")
        print(f"   📚 Total vocabulary analyses: {summary['total_vocabulary_analyses']}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()