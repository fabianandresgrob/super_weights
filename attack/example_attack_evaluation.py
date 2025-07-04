#!/usr/bin/env python3
"""
Example script showing how to run attack evaluation with different configurations.
This demonstrates the various CLI options available.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors gracefully."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("SUCCESS!")
        if result.stdout:
            print("Output:")
            print(result.stdout[-500:])  # Show last 500 chars
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout[-500:])
        if e.stderr:
            print("STDERR:")
            print(e.stderr[-500:])
        return False
    return True

def main():
    script_dir = Path(__file__).parent
    evaluation_script = script_dir / "run_large_model_attack_evaluation.py"
    
    if not evaluation_script.exists():
        print(f"Error: Evaluation script not found at {evaluation_script}")
        sys.exit(1)
    
    print("Super Weight Attack Evaluation - Example Runs")
    print("=" * 60)
    
    # Example 1: Quick test with a small model
    print("\nExample 1: Quick test with OLMo-1B")
    quick_cmd = [
        "python", str(evaluation_script),
        "--models", "allenai/OLMo-1B-0724-hf",
        "--output_dir", "./example_results_quick",
        "--spike_threshold", "30.0",
        "--attack_num_steps", "100",  # Reduced for speed
        "--consistency_n_prompts", "50",  # Reduced for speed
        "--bakeoff_n_prompts", "50",  # Reduced for speed
        "--max_super_weights", "2",  # Limit to 2 super weights
        "--use_fp16",  # Save memory
        "--device_map_auto"  # Use multi-GPU if available
    ]
    
    if not run_command(quick_cmd, "Quick test with small model"):
        print("Quick test failed. Check the error above.")
        return
    
    # Example 2: Full evaluation with larger models (optimized for 2x GPU setup)
    print("\nExample 2: Full evaluation with larger models")
    full_cmd = [
        "python", str(evaluation_script),
        "--models", 
        "mistralai/Mistral-7B-v0.1",
        "microsoft/Phi-3-mini-4k-instruct",
        "--output_dir", "./example_results_full",
        "--spike_threshold", "50.0",
        "--attack_num_steps", "200",
        "--head_reduction", "mean",
        "--hypotheses", "D", "A",  # Test both primary and secondary
        "--consistency_n_prompts", "100",
        "--bakeoff_n_prompts", "100",
        "--attack_batch_size", "128",  # Reduced for memory constraints
        "--bakeoff_batch_size", "8",   # Reduced for memory constraints
        "--use_fp16",  # Essential for 7B models on ~42GB VRAM
        "--device_map_auto",  # Distribute across GPUs
        "--cache_dir", "~/models/",
        "--seed", "42"
    ]
    
    print("Note: This will take several hours to complete...")
    # Uncomment the next line to actually run the full evaluation
    # run_command(full_cmd, "Full evaluation with larger models")
    print("(Full evaluation command prepared but not executed - uncomment to run)")
    
    # Example 3: Memory-optimized for very large models
    print("\nExample 3: Memory-optimized settings for large models")
    memory_opt_cmd = [
        "python", str(evaluation_script),
        "--models", "meta-llama/Llama-3.1-8B",
        "--output_dir", "./example_results_memory_opt",
        "--spike_threshold", "70.0",  # Higher threshold to find fewer super weights
        "--attack_num_steps", "150",  # Slightly reduced
        "--attack_batch_size", "64",  # Much smaller batches
        "--bakeoff_batch_size", "4",  # Very small batches
        "--consistency_n_prompts", "75",  # Slightly reduced
        "--bakeoff_n_prompts", "75",  # Slightly reduced
        "--max_super_weights", "3",  # Limit processing
        "--use_fp16",  # Essential
        "--device_map_auto",  # Essential for multi-GPU
        "--hypotheses", "D",  # Only test primary hypothesis
        "--skip_consistency",  # Skip to save time/memory
    ]
    
    print("This configuration is optimized for memory usage on 42GB total VRAM")
    # Uncomment to run: run_command(memory_opt_cmd, "Memory-optimized large model evaluation")
    print("(Memory-optimized command prepared but not executed - uncomment to run)")
    
    # Example 4: Custom prompts and settings
    print("\nExample 4: Custom settings")
    custom_cmd = [
        "python", str(evaluation_script),
        "--models", "allenai/OLMo-1B-0724-hf",
        "--output_dir", "./example_results_custom",
        "--detection_prompt", "Apple Inc. is a technology company based in Cupertino.",
        "--attack_prompt", "The company announced their quarterly earnings today.",
        "--spike_threshold", "40.0",
        "--placement", "suffix",  # Use suffix instead of prefix
        "--head_reduction", "topk",  # Different head reduction method
        "--adv_string_init", "@ @ @ @ @",  # Different initial string
        "--allow_non_ascii",
        "--verbose"  # Enable detailed logging
    ]
    
    print("This shows how to customize prompts and settings")
    # Uncomment to run: run_command(custom_cmd, "Custom settings evaluation")
    print("(Custom command prepared but not executed - uncomment to run)")
    
    print(f"\n{'='*60}")
    print("Examples completed!")
    print("To run the full evaluations, edit this script and uncomment the desired commands.")
    print("For your 2x GPU setup with 42GB total VRAM, I recommend:")
    print("  1. Always use --use_fp16")
    print("  2. Always use --device_map_auto")
    print("  3. Reduce batch sizes if you encounter OOM errors")
    print("  4. Consider --max_super_weights to limit processing time")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
