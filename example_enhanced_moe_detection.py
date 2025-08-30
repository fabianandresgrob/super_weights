#!/usr/bin/env python3
"""
Enhanced MoE Super Weight Detection Example

This script demonstrates the new enhanced MoE super weight detection capabilities
that implement the mathematical framework from the prompt, including:

1. Routing statistics: p_active^(l)(e), position-wise entropy H^(l)(pos)
2. Per-expert co-spike detection: S^(l,e)(r,c) alignment scores
3. Causal impact scoring: I_nat and I_int with natural vs interventional routing
4. Fast proxy metrics: energy reduction and stopword skew

Usage:
    python example_enhanced_moe_detection.py --model "microsoft/Phi-3-mini-4k-instruct"
    python example_enhanced_moe_detection.py --model "mistralai/Mixtral-8x7B-v0.1" --enable-causal
"""

import argparse
import logging
import json
import torch
from typing import Dict, Any, List

# Setup path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from research.researcher import SuperWeightResearchSession
from detection.super_weight import MoESuperWeight


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def enhanced_moe_detection_demo(model_name: str, 
                              input_text: str,
                              enable_causal_scoring: bool = False,
                              p_active_floor: float = 0.01,
                              co_spike_threshold: float = 0.12) -> Dict[str, Any]:
    """
    Demonstrate enhanced MoE super weight detection
    
    Args:
        model_name: HuggingFace model name or path
        input_text: Text for detection
        enable_causal_scoring: Whether to compute causal impact scores
        p_active_floor: Minimum p_active for expert consideration
        co_spike_threshold: Co-spike alignment threshold
    """
    print("Enhanced MoE Super Weight Detection Demo")
    print(f"Model: {model_name}")
    print(f"Co-spike threshold: {co_spike_threshold}")
    print(f"p_active floor: {p_active_floor}")
    print(f"Causal scoring: {'✓' if enable_causal_scoring else '✗'}")
    print("-" * 60)
    
    # Initialize research session
    session = SuperWeightResearchSession.from_model_name(
        model_name, 
        model_kwargs={'torch_dtype': torch.float16, 'device_map': 'auto'},
        log_level=logging.INFO
    )

    print(f"Model Info:")
    print(f"  Architecture: {session.model_info['architecture']}")
    print(f"  MoE Model: {'Yes' if session.model_info['is_moe'] else 'No'}")
    print(f"  Layers: {session.model_info['num_layers']}")
    if session.model_info['is_moe']:
        print(f"  MoE Layers: {session.model_info['moe_layers']}")
    print()
    
    if not session.model_info['is_moe']:
        print("This model is not MoE. Enhanced MoE detection requires a MoE model.")
        print("   Try models like: microsoft/Phi-3-mini-4k-instruct, mistralai/Mixtral-8x7B-v0.1")
        return {'error': 'Not a MoE model'}
    
    # Phase 1: Enhanced MoE Detection
    print("Phase 1: Enhanced MoE Super Weight Detection")
    super_weights = session.detect_super_weights(
        input_text=input_text,
        router_analysis_samples=8,
        p_active_floor=p_active_floor,
        co_spike_threshold=co_spike_threshold,
        enable_causal_scoring=enable_causal_scoring,
        max_iterations=8
    )
    
    print(f"Found {len(super_weights)} MoE super weights")
    print()
    
    if not super_weights:
        print("No super weights detected. Try adjusting thresholds or input text.")
        return {'super_weights': [], 'analysis': {}}
    
    # Phase 2: Analyze Results
    print("Phase 2: Enhanced MoE Analysis")
    analysis_results = analyze_moe_super_weights(super_weights, enable_causal_scoring)
    
    # Phase 3: Display Results
    print("Phase 3: Results Summary")
    display_moe_results(super_weights, analysis_results)
    
    return {
        'model_info': session.model_info,
        'super_weights': [serialize_moe_super_weight(sw) for sw in super_weights],
        'analysis': analysis_results,
        'detection_config': {
            'p_active_floor': p_active_floor,
            'co_spike_threshold': co_spike_threshold,
            'enable_causal_scoring': enable_causal_scoring
        }
    }


def analyze_moe_super_weights(super_weights: List[MoESuperWeight], 
                             enable_causal_scoring: bool) -> Dict[str, Any]:
    """Analyze patterns in detected MoE super weights"""
    
    analysis = {
        'total_count': len(super_weights),
        'layer_distribution': {},
        'expert_distribution': {},
        'routing_statistics': {},
        'co_spike_statistics': {},
        'causal_analysis': {} if enable_causal_scoring else None
    }
    
    # Layer distribution
    layer_counts = {}
    expert_counts = {}
    p_actives = []
    co_spike_scores = []
    
    for sw in super_weights:
        # Layer distribution
        layer_counts[sw.layer] = layer_counts.get(sw.layer, 0) + 1
        
        # Expert distribution  
        expert_key = f"L{sw.layer}E{sw.expert_id}"
        expert_counts[expert_key] = expert_counts.get(expert_key, 0) + 1
        
        # Routing statistics
        if sw.p_active is not None:
            p_actives.append(sw.p_active)
            
        # Co-spike statistics
        if sw.co_spike_score is not None:
            co_spike_scores.append(sw.co_spike_score)
    
    analysis['layer_distribution'] = dict(sorted(layer_counts.items()))
    analysis['expert_distribution'] = dict(sorted(expert_counts.items()))
    
    # Routing statistics
    if p_actives:
        analysis['routing_statistics'] = {
            'mean_p_active': sum(p_actives) / len(p_actives),
            'min_p_active': min(p_actives),
            'max_p_active': max(p_actives),
            'total_experts_analyzed': len(set(f"L{sw.layer}E{sw.expert_id}" for sw in super_weights))
        }
    
    # Co-spike statistics
    if co_spike_scores:
        analysis['co_spike_statistics'] = {
            'mean_score': sum(co_spike_scores) / len(co_spike_scores),
            'min_score': min(co_spike_scores),
            'max_score': max(co_spike_scores),
            'scores_above_threshold': len([s for s in co_spike_scores if s > 0.15])
        }
    
    # Causal analysis
    if enable_causal_scoring:
        natural_impacts = [sw.impact_natural for sw in super_weights if sw.impact_natural is not None]
        interventional_impacts = [sw.impact_interventional for sw in super_weights if sw.impact_interventional is not None]
        causal_agreements = [sw.causal_agreement for sw in super_weights if sw.causal_agreement is not None]
        
        if natural_impacts:
            analysis['causal_analysis'] = {
                'mean_natural_impact': sum(natural_impacts) / len(natural_impacts),
                'mean_interventional_impact': sum(interventional_impacts) / len(interventional_impacts) if interventional_impacts else 0,
                'mean_causal_agreement': sum(causal_agreements) / len(causal_agreements) if causal_agreements else 0,
                'strong_agreement_count': len([a for a in causal_agreements if abs(a - 1.0) < 0.3])
            }
    
    return analysis


def display_moe_results(super_weights: List[MoESuperWeight], analysis: Dict[str, Any]):
    """Display MoE super weight results in a readable format"""

    print(f"Top Super Weights (showing up to 10):")
    print("-" * 80)
    
    # Sort by co-spike score if available, otherwise by magnitude
    sorted_weights = sorted(super_weights, 
                          key=lambda sw: sw.co_spike_score if sw.co_spike_score else sw.magnitude_product,
                          reverse=True)
    
    for i, sw in enumerate(sorted_weights[:10]):
        print(f"{i+1:2d}. Layer {sw.layer}, Expert {sw.expert_id}: [{sw.row}, {sw.column}]")
        if sw.p_active is not None:
            print(f"    p_active: {sw.p_active:.4f}")
        if sw.co_spike_score is not None:
            print(f"    Co-spike score: {sw.co_spike_score:.4f}")
        if sw.routed_tokens_count is not None:
            print(f"    Routed tokens: {sw.routed_tokens_count}")
        if sw.impact_natural is not None:
            print(f"    Impact (natural): {sw.impact_natural:.4f}")
        if sw.impact_interventional is not None:
            print(f"    Impact (interventional): {sw.impact_interventional:.4f}")
        if sw.causal_agreement is not None:
            print(f"    Causal agreement: {sw.causal_agreement:.3f}")
        print()
    
    print("Analysis Summary:")
    print("-" * 40)
    print(f"Layer distribution: {analysis['layer_distribution']}")
    print(f"Expert distribution: {len(analysis['expert_distribution'])} unique expert instances")
    
    if analysis['routing_statistics']:
        stats = analysis['routing_statistics']
        print(f"Routing stats: mean p_active = {stats['mean_p_active']:.4f}")
        print(f"               range = [{stats['min_p_active']:.4f}, {stats['max_p_active']:.4f}]")
    
    if analysis['co_spike_statistics']:
        stats = analysis['co_spike_statistics']
        print(f"Co-spike stats: mean = {stats['mean_score']:.4f}")
        print(f"                range = [{stats['min_score']:.4f}, {stats['max_score']:.4f}]")
    
    if analysis.get('causal_analysis'):
        stats = analysis['causal_analysis']
        print(f"Causal analysis: mean natural impact = {stats['mean_natural_impact']:.4f}")
        print(f"                 mean agreement = {stats['mean_causal_agreement']:.3f}")
        print(f"                 strong agreement count = {stats['strong_agreement_count']}")


def serialize_moe_super_weight(sw: MoESuperWeight) -> Dict[str, Any]:
    """Serialize MoESuperWeight for JSON output"""
    return {
        'layer': sw.layer,
        'expert_id': sw.expert_id,
        'row': sw.row,
        'column': sw.column,
        'component': sw.component,
        'input_value': sw.input_value,
        'output_value': sw.output_value,
        'magnitude_product': sw.magnitude_product,
        'p_active': sw.p_active,
        'co_spike_score': sw.co_spike_score,
        'routed_tokens_count': sw.routed_tokens_count,
        'impact_natural': sw.impact_natural,
        'impact_interventional': sw.impact_interventional,
        'causal_agreement': sw.causal_agreement,
        'energy_reduction': sw.energy_reduction,
        'stopword_skew': sw.stopword_skew
    }


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced MoE Super Weight Detection Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic MoE detection
  python example_enhanced_moe_detection.py --model "microsoft/Phi-3-mini-4k-instruct"
  
  # With causal scoring
  python example_enhanced_moe_detection.py --model "mistralai/Mixtral-8x7B-v0.1" --enable-causal
  
  # Custom thresholds
  python example_enhanced_moe_detection.py --model "microsoft/Phi-3-mini-4k-instruct" \\
    --co-spike-threshold 0.15 --p-active-floor 0.02
        """
    )
    
    parser.add_argument(
        '--model', '-m', 
        required=True,
        help='Model name or path (should be MoE model)'
    )
    
    parser.add_argument(
        '--input-text', '-t',
        default="Apple Inc. is a worldwide technology company that designs consumer electronics.",
        help='Input text for detection (default: Apple Inc. sentence)'
    )
    
    parser.add_argument(
        '--co-spike-threshold', '-c',
        type=float, default=0.12,
        help='Co-spike alignment score threshold (default: 0.12)'
    )
    
    parser.add_argument(
        '--p-active-floor', '-p',
        type=float, default=0.01,
        help='Minimum p_active for expert consideration (default: 0.01)'
    )
    
    parser.add_argument(
        '--enable-causal', '-e',
        action='store_true',
        help='Enable causal impact scoring (slower but more informative)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output JSON file for results'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Run demo
        results = enhanced_moe_detection_demo(
            model_name=args.model,
            input_text=args.input_text,
            enable_causal_scoring=args.enable_causal,
            p_active_floor=args.p_active_floor,
            co_spike_threshold=args.co_spike_threshold
        )
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
        
        print("\nEnhanced MoE detection demo completed successfully!")
        
    except Exception as e:
        print(f"\nError during detection: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
