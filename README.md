# Super Weights Research Framework

A comprehensive Python framework for detecting, analyzing, and understanding **super weights** in Large Language Models. Super weights are individual parameters that have disproportionate impact on model performance - despite being tiny fractions of total parameters, removing even a single super weight can catastrophically degrade model performance.

## 🔬 What are Super Weights?

Super weights are individual parameters in transformer models that:
- **Cause massive activations** (thousands of times larger than median values)
- **Have catastrophic impact** when removed (accuracy drops to guessing levels) 
- **Are sparsely distributed** across layers and components
- **Challenge the "no privileged basis" assumption** in transformers

This framework implements detection methods from recent research and provides comprehensive analysis tools for understanding these critical parameters.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone [repository-url]
cd super_weights

# Create conda environment
conda env create -f environment.yml
conda activate super-weights
```

### Basic Usage

```python
from research.researcher import SuperWeightResearchSession

# Initialize research session with any supported model
session = SuperWeightResearchSession.from_model_name('allenai/OLMo-1B-0724-hf')

# Detect super weights
# For MoE models you can optionally control router analysis samples
super_weights = session.detect_super_weights(spike_threshold=70.0, router_analysis_samples=5)
print(f"Found {len(super_weights)} super weights")

# Analyze vocabulary effects
vocab_analysis = session.analyzer.vocabulary_analyzer.analyze_vocabulary_effects(super_weights[0])
session.analyzer.vocabulary_analyzer.display_analysis_results(vocab_analysis)

# Measure impact on model performance
perplexity_impact = session.analyzer.metrics_analyzer.measure_perplexity_impact(super_weights[0])
accuracy_impact = session.analyzer.metrics_analyzer.measure_accuracy_impact(super_weights[0])
```

### Interactive Analysis

See [`validate_super_weights.ipynb`](validate_super_weights.ipynb) for a complete interactive analysis example including:
- Super weight detection with iterative refinement
- Vocabulary effect analysis showing token-level impacts
- Cascade analysis through network layers
- Perplexity and accuracy impact measurement
- Generation experiments with manipulated super weights

## 📁 Repository Structure

```
super_weights/
├── detection/          # Core detection algorithms
│   ├── detector.py     # SuperWeightDetector class
│   └── super_weight.py # SuperWeight dataclass
├── management/         # Safe weight manipulation
│   └── manager.py      # SuperWeightManager with context managers
├── analysis/           # Comprehensive analysis tools
│   ├── analyzer.py     # Main analysis coordinator
│   ├── vocabulary.py   # Vocabulary effect analysis
│   ├── metrics.py      # Perplexity/accuracy measurement
│   ├── patterns.py     # Spatial/functional pattern analysis
│   └── activation.py   # Activation source tracing
├── research/           # High-level research workflows
│   └── researcher.py   # SuperWeightResearchSession
├── utils/              # Model architecture support
│   ├── model_architectures.py  # Universal model handlers
│   ├── datasets.py     # Dataset loading utilities
│   └── model_list.json # Supported models
├── scripts/            # Batch analysis scripts
└── results/            # Analysis results and plots
```

## 🎯 Key Features

### 🔍 **Multi-Architecture Support**
- **Standard Models**: LLaMA, Mistral, OLMo, Phi-3, GPT-style models
- **Universal Detection**: Automatic architecture detection and adaptation
- **Flexible Components**: Works with different MLP structures and configurations

### 📊 **Comprehensive Analysis**
- **Vocabulary Effects**: How super weights influence token probabilities
- **Cascade Analysis**: Activation propagation through network layers
- **Metrics Impact**: Perplexity and accuracy degradation measurement
- **Pattern Analysis**: Spatial and functional pattern detection
- **Activation Tracing**: Source analysis of massive activations

### 🛡️ **Safe Experimentation**
- **Context Managers**: Automatic weight backup/restore
- **Temporary Modifications**: Safe weight scaling/zeroing
- **State Tracking**: Complete modification history
- **Error Recovery**: Automatic restoration on exceptions

### 🔬 **Research Workflows**
- **Detection Pipeline**: Iterative super weight identification
- **Screening Analysis**: Impact-based ranking of super weights
- **Batch Processing**: Multi-model comparative analysis
- **Session Management**: Reproducible research with saved sessions

## 📚 Supported Models

### Currently Tested Models
- **LLaMA Family**: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-3.1-8B`
- **Mistral**: `mistralai/Mistral-7B-v0.1`
- **OLMo**: `allenai/OLMo-1B-0724-hf`, `allenai/OLMo-7B-0724-hf`
- **Phi-3**: `microsoft/Phi-3-mini-4k-instruct`

*See [`utils/model_list.json`](utils/model_list.json) for complete list*

## 🧪 Usage Examples

### Detection and Basic Analysis

```python
from research.researcher import SuperWeightResearchSession

# Initialize session
session = SuperWeightResearchSession.from_model_name('mistralai/Mistral-7B-v0.1')

# Detect super weights
super_weights = session.detect_super_weights(
    input_text="Apple Inc. is a worldwide tech company.",
    spike_threshold=50.0,
    max_iterations=10,
    router_analysis_samples=5  # Only used for MoE models
)

# Quick screening to find most impactful super weights
screening_results = session.quick_screening(super_weights)
print(f"Most impactful: {screening_results[0]['super_weight']}")
```

### Vocabulary Effect Analysis

```python
# Analyze how super weight affects vocabulary
vocab_analysis = session.analyzer.vocabulary_analyzer.analyze_vocabulary_effects(
    super_weights[0], 
    n_samples=500
)

# Display results with visualizations
session.analyzer.vocabulary_analyzer.display_analysis_results(vocab_analysis, top_k=20)

# Analyze activation cascade through layers
cascade_results = session.analyzer.vocabulary_analyzer.analyze_activation_cascade(
    super_weights[0],
    input_text="The company develops innovative technology."
)
```

### Safe Weight Manipulation

```python
# Temporarily zero out super weights
with session.manager.temporary_zero([super_weights[0]]):
    # Model now has super weight zeroed
    output = session.model.generate(tokens, max_new_tokens=10)
    print("Generated with super weight zeroed:", output)
# Super weight automatically restored here

# Scale super weights
with session.manager.temporary_scale([super_weights[0]], scale_factor=2.0):
    # Super weight is now doubled
    output = session.model.generate(tokens, max_new_tokens=10)
    print("Generated with super weight doubled:", output)
```

### Impact Measurement

```python
# Measure perplexity impact
perplexity_results = session.analyzer.metrics_analyzer.measure_perplexity_impact(
    super_weights[0],
    dataset_name='wikitext',
    n_samples=500
)

# Measure accuracy impact
accuracy_results = session.analyzer.metrics_analyzer.measure_accuracy_impact(
    super_weights[0],
    task='hellaswag',
    n_samples=200
)

print(f"Perplexity: {perplexity_results['baseline_perplexity']:.2f} → {perplexity_results['modified_perplexity']:.2f}")
print(f"Accuracy: {accuracy_results['baseline_accuracy']:.2%} → {accuracy_results['modified_accuracy']:.2%}")
```

### Full Research Pipeline

```python
# Run complete analysis pipeline
results = session.full_research_pipeline(
    detection_config={
        'spike_threshold': 50.0,
        'max_iterations': 10
    },
    analysis_config={
        'max_detailed': 3,
        'analysis_types': ['screen', 'vocabulary', 'metrics', 'patterns']
    }
)

# Export results for reproducibility
session.export_session('my_analysis_results.json')
```

## 🔧 Advanced Usage

### Custom Detection Parameters

```python
# Fine-tune detection for specific models
from detection.detector import SuperWeightDetector

detector = SuperWeightDetector(model, tokenizer)
super_weights = detector.detect_super_weights(
    input_text="Custom prompt for your domain",
    spike_threshold=30.0,  # Lower threshold for more sensitivity
    max_iterations=15      # More iterations for thorough search
)
```

### Activation Source Tracing

```python
# Trace the source of massive activations
trace_result = session.analyzer.trace_activation_source(super_weights[0])
print(f"Pathway: {trace_result['pathway_analysis']['super_weight_location']}")
print(f"Key findings: {trace_result['pathway_analysis']['key_findings']}")
print(f"Total amplification: {trace_result['computational_flow']['amplification_factors']['total_amplification']:.1f}x")
```

### Mathematical Analysis

```python
# Analyze the mathematical properties of super weights
math_analysis = session.analyzer.mathematical_super_activation_analysis(super_weights[0])
print("Mathematical breakdown:", math_analysis['mathematical_breakdown'])
print("Weight analysis:", math_analysis['weight_analysis'])
print("Attack vectors:", math_analysis['attack_vectors'])
```

## 🎯 Goals and Future Work

### 🔮 **Planned Extensions**
- **Mixture of Experts**: Support for Mixtral, OLMoE, DeepSeek-MoE, Qwen-MoE models. See [MOE_SUPER_WEIGHT_MIGRATION.md](MOE_SUPER_WEIGHT_MIGRATION.md) for the routing-aware migration guide.
- **Cross-Model Analysis**: Comparative studies across different architectures
- **Intervention Strategies**: Advanced techniques for super weight manipulation
- **Theoretical Analysis**: Mathematical frameworks for understanding super weights

### 🌟 **Research Directions**
- **Universality**: Whether super weights exist across all transformer architectures
- **Transferability**: How super weights relate across different model sizes and training
- **Robustness**: Understanding model vulnerability through super weight analysis
- **Optimization**: Leveraging super weight knowledge for model compression and efficiency

## 📖 Related Research

This framework implements and extends methods from:
- **Super Weights Paper** (2024): Original super weight detection methodology
- **Massive Activations Research**: Investigation of extreme activation phenomena
- **Privileged Basis Analysis**: Challenges to transformer theoretical assumptions

## 🔬 Research Applications

This framework enables research into:
- **Interpretability**: Understanding transformer internal mechanisms
- **Robustness**: Analyzing model vulnerability to targeted attacks
- **Efficiency**: Identifying critical parameters for model compression
- **Safety**: Understanding model behavior modification mechanisms