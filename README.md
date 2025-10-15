# Super Weights Research Framework

A comprehensive Python framework for detecting, analyzing, and understanding **super weights** in Large Language Models. Super weights are individual parameters that have disproportionate impact on model performance - despite being tiny fractions of total parameters, removing even a single super weight can catastrophically degrade model performance.

## 🔬 What are Super Weights?

Super weights are individual parameters in transformer models that:
- **Cause massive activations** (thousands of times larger than median values)
- **Have catastrophic impact** when removed (accuracy drops to guessing levels) 
- **Are sparsely distributed** across layers and components
- **Challenge the "no privileged basis" assumption** in transformers

This framework implements detection methods from [Yu et al. (2025)](https://arxiv.org/abs/2411.07191) and provides comprehensive analysis tools for understanding these critical parameters. Vocabulary analysis components build upon the Universal Neurons framework from [Bills et al. (2023)](https://arxiv.org/abs/2401.12181).

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
super_weights = session.detect_super_weights(spike_threshold=70.0)
print(f"Found {len(super_weights)} super weights")

# Analyze vocabulary effects with Universal Neurons processing
vocab_analysis = session.analyzer.vocabulary_analyzer.analyze_neuron_vocabulary_effects(
    super_weights[0], 
    apply_universal_neurons_processing=True
)
session.analyzer.vocabulary_analyzer.display_vocabulary_card(vocab_analysis)

# Measure impact on model performance
perplexity_impact = session.analyzer.metrics_analyzer.measure_perplexity_impact(super_weights[0])
accuracy_impact = session.analyzer.metrics_analyzer.measure_accuracy_impact(super_weights[0])
```

### Interactive Analysis

See [`validate_super_weights.ipynb`](validate_super_weights.ipynb) for a complete interactive analysis example including:
- Super weight detection with iterative refinement
- Vocabulary effect analysis with Universal Neurons processing
- Cascade analysis through network layers showing activation propagation
- Intervention analysis with robust windowed evaluation
- Control experiments comparing against random baselines
- Token class enrichment analysis
- Perplexity and accuracy impact measurement
- Mathematical breakdown of super activation pathways

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
│   ├── vocabulary.py   # Vocabulary effect analysis with Universal Neurons
│   ├── metrics.py      # Perplexity/accuracy measurement
│   ├── activation.py   # Mathematical super activation analysis
│   └── head_analyzer.py # Attention head analysis for attacks
├── attack/             # Adversarial attack capabilities
│   ├── attack.py       # GCG-style adversarial attacks
│   └── attack_eval.py  # Attack evaluation and metrics
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
- **Mixture of Experts**: Qwen1.5-MoE, DeepSeek-MoE, OLMoE, Mixtral, Phi-3.5-MoE
- **Hybrid Architectures**: Models combining regular MLP and MoE layers (e.g., DeepSeek-V2)
- **Universal Detection**: Automatic architecture detection and adaptation
- **Flexible Components**: Works with different MLP structures and configurations

### 📊 **Comprehensive Analysis**
- **Vocabulary Effects**: Token probability analysis with Universal Neurons processing
- **Cascade Analysis**: Activation propagation through network layers with configurable depth
- **Intervention Analysis**: Robust windowed evaluation of super weight removal
- **Control Experiments**: Random baseline comparisons and deterministic validation
- **Token Class Enrichment**: Analysis of effects on semantic token categories
- **Mathematical Analysis**: Detailed breakdown of super activation pathways
- **Metrics Impact**: Perplexity and accuracy degradation measurement
- **Pattern Analysis**: Spatial and functional pattern detection

### 🛡️ **Safe Experimentation**
- **Context Managers**: Automatic weight backup/restore
- **Temporary Modifications**: Safe weight scaling/zeroing
- **State Tracking**: Complete modification history
- **Error Recovery**: Automatic restoration on exceptions

### 🔬 **Research Workflows**
- **Detection Pipeline**: Iterative super weight identification with MoE routing analysis
- **Screening Analysis**: Impact-based ranking of super weights
- **Batch Processing**: Multi-model comparative analysis
- **Session Management**: Reproducible research with saved sessions
- **Attack Capabilities**: GCG-style adversarial attacks targeting super weights

## 📚 Supported Models

### Currently Tested Models
- **LLaMA Family**: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-3.1-8B`, `meta-llama/Llama-3.2-3B`
- **Mistral**: `mistralai/Mistral-7B-v0.1`
- **OLMo**: `allenai/OLMo-1B-0724-hf`, `allenai/OLMo-7B-0724-hf`
- **Phi-3**: `microsoft/Phi-3-mini-4k-instruct`, `microsoft/phi-4`
- **Mixture of Experts**: `Qwen/Qwen1.5-MoE-A2.7B`, `deepseek-ai/deepseek-moe-16b-base`, `allenai/OLMoE-1B-7B-0924`


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
    max_iterations=10
)

# Quick screening to find most impactful super weights
screening_results = session.quick_screening(super_weights)
print(f"Most impactful: {screening_results[0]['super_weight']}")
```

### Vocabulary Effect Analysis

```python
# Analyze how super weight affects vocabulary with Universal Neurons processing
vocab_analysis = session.analyzer.vocabulary_analyzer.analyze_neuron_vocabulary_effects(
    super_weights[0], 
    apply_universal_neurons_processing=True
)

# Display results with enhanced insights
session.analyzer.vocabulary_analyzer.display_vocabulary_card(vocab_analysis, top_k_display=10)

# Analyze intervention effects with robust windowed evaluation
intervention_results = session.analyzer.vocabulary_analyzer.analyze_super_weight_intervention(
    super_weights[0],
    n_samples=100
)

# Analyze cascade effects through layers
cascade_results = session.analyzer.vocabulary_analyzer.analyze_cascade_effects(
    super_weights[0],
    input_text="Apple Inc. develops innovative technology solutions.",
    num_layers=8,
    top_k_margin=15
)

# Run comprehensive control experiments
control_results = session.analyzer.vocabulary_analyzer.analyze_controls_and_baselines(
    super_weights[0],
    n_samples=100
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
from detection.detector import SuperWeightDetector, MoESuperWeightDetector

# For standard models
detector = SuperWeightDetector(model, tokenizer, mlp_handler, manager)
super_weights = detector.detect_super_weights(
    input_text="Custom prompt for your domain",
    spike_threshold=30.0,  # Lower threshold for more sensitivity
    max_iterations=15      # More iterations for thorough search
)

# For MoE models with advanced parameters
moe_detector = MoESuperWeightDetector(model, tokenizer, mlp_handler, architecture_type)
super_weights = moe_detector.detect_super_weights(
    router_analysis_samples=10,
    p_active_floor=0.005,  # Lower floor for rare experts
    co_spike_threshold=0.15,
    enable_causal_scoring=True
)
```

### Enhanced Vocabulary Analysis

```python
# Compare enhanced vs standard analysis
enhanced_analysis = session.analyzer.vocabulary_analyzer.analyze_neuron_vocabulary_effects(
    super_weights[0], 
    apply_universal_neurons_processing=True
)

standard_analysis = session.analyzer.vocabulary_analyzer.analyze_neuron_vocabulary_effects(
    super_weights[0], 
    apply_universal_neurons_processing=False
)

# Token class enrichment analysis
enrichment = enhanced_analysis['raw_analysis']['enrichment']
print(f"Best enriched class: {enrichment['best_theme']['class']}")
print(f"Enrichment score: {enrichment['best_theme']['score']:.3f}")
```

### Mathematical Analysis

```python
# Detailed mathematical breakdown of super activation
math_analysis = session.analyzer.mathematical_super_activation_analysis(super_weights[0])
print("Mathematical breakdown:", math_analysis['mathematical_breakdown'])
print("Weight analysis:", math_analysis['weight_analysis'])
```

### Adversarial Attacks

```python
# Target super weights with adversarial attacks (see attack/ folder for details)
from attack.attack import SuperWeightAttacker, SuperWeightAttackConfig

# Configure attack targeting a specific super weight
config = SuperWeightAttackConfig(
    target=super_weights[0],
    hypothesis='A',  # Attack hypothesis (A-E available)
    num_steps=500,
    learning_rate=0.01
)

# Run GCG-style attack to functionally disable the super weight
attacker = SuperWeightAttacker(model, tokenizer, config)
attack_results = attacker.attack()
print(f"Attack success: {attack_results['final_adv_string']}")
```

## 🎯 Advanced Capabilities

### 🤖 **Mixture of Experts Support**
- **Multi-Expert Detection**: Specialized algorithms for MoE models (Qwen1.5-MoE, DeepSeek-MoE, OLMoE, Mixtral)
- **Routing Analysis**: Expert activation probability analysis and entropy-based candidate selection
- **Hybrid Architecture**: Support for models with both regular MLP and MoE layers
- **Co-spike Detection**: Advanced detection algorithms for expert-specific super weights

### 🎯 **Adversarial Attack Framework**
- **GCG-Style Attacks**: Gradient-based optimization for targeting super weights
- **Multiple Hypotheses**: Various attack strategies (A-E) for different super weight mechanisms
- **Attention Sink Attacks**: Specialized attacks on attention-based super weight pathways
- **Multi-Prompt Optimization**: Robust adversarial string generation across multiple contexts

### 🧮 **Mathematical Analysis Tools**
- **Activation Pathway Tracing**: Mathematical breakdown of super activation computation
- **Architecture-Specific Analysis**: Supports standard, gated, and fused MLP architectures
- **Universal Neurons Integration**: Layer norm folding and mean-centering for robust analysis
- **Statistical Validation**: Bootstrap confidence intervals and deterministic verification

### 📊 **Comprehensive Evaluation**
- **Token Class Enrichment**: Effects on digits, punctuation, function words, etc.
- **Cascade Effect Analysis**: Layer-by-layer propagation with configurable sampling
- **Control Baselines**: Random coordinate comparisons and full neuron ablation
- **Intervention Metrics**: Loss, entropy, top-k margin, and stopword mass analysis

## 📖 Related Research

This framework implements and extends methods from:
- **Yu et al. (2025)**: [The Super Weight in Large Language Models](https://arxiv.org/abs/2411.07191) - Original super weight detection methodology
- **Gurnee et al. (2024)**: [Universal Neurons in GPT2 Language Models](https://arxiv.org/abs/2401.12181) - Universal Neurons framework for neuron analysis

## 🔬 Research Applications

This framework enables research into:
- **Interpretability**: Understanding transformer internal mechanisms through super weight analysis
- **Robustness**: Analyzing model vulnerability to targeted adversarial attacks
- **Efficiency**: Identifying critical parameters for model compression and optimization
- **Safety**: Understanding model behavior modification mechanisms
- **Architecture Analysis**: Comparative studies of standard vs. MoE model vulnerabilities
- **Universal Phenomena**: Cross-model investigation of super weight prevalence and characteristics

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- ~16GB GPU memory for 7B models, ~8GB for 1B models

### Installation
```bash
# Clone the repository
git clone [repository-url]
cd super_weights

# Create conda environment
conda env create -f environment.yml
conda activate super-weights
```

### Quick Example
```python
from research.researcher import SuperWeightResearchSession

# Start with a smaller model for quick testing
session = SuperWeightResearchSession.from_model_name('allenai/OLMo-1B-0724-hf')
super_weights = session.detect_super_weights(spike_threshold=70.0)
vocab_analysis = session.analyzer.vocabulary_analyzer.analyze_neuron_vocabulary_effects(super_weights[0])
session.analyzer.vocabulary_analyzer.display_vocabulary_card(vocab_analysis)
```

## 📄 Documentation

- **Main Analysis**: [`validate_super_weights.ipynb`](validate_super_weights.ipynb) - Comprehensive examples
- **Attack Framework**: [`attack/README.md`](attack/README.md) - Adversarial attack documentation  
- **Scripts**: [`scripts/README.md`](scripts/README.md) - Batch analysis tools
- **API Reference**: Docstrings throughout the codebase provide detailed parameter documentation
