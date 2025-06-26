# Refresh notes
You must maintain a "CLAUDE.md refresh counter" that starts at 10.
Decrement this counter by 1 after each of your responses to me.
At the end of each response, explicitly state "CLAUDE.md refresh counter: [current value]".
When the counter reaches 1, you MUST include the entire contents of CLAUDE.md in your next response to refresh your memory,
and then reset the counter to 10.

# Super Weights Project Guidelines

## Project Structure & Architecture

### Core Package Structure
- **`super_weights/`** - Main package for super weights analysis
  - **`detection/`** - Core detection logic and data structures
    - `detector.py` - Pure detection algorithms for identifying super weights
    - `super_weight.py` - SuperWeight dataclass definition and utilities
  - **`management/`** - Weight modification and manipulation
    - `manager.py` - Context managers for safe weight modification experiments
  - **`analysis/`** - Analysis modules for understanding super weight behavior
    - `analyzer.py` - Main analyzer coordinator orchestrating different analysis types
    - `vocabulary.py` - Universal Neurons adapted methods for linguistic analysis
    - `metrics.py` - Perplexity and accuracy measurement utilities
    - `patterns.py` - Pattern detection and analysis for super weight characteristics
  - **`research/`** - High-level research workflows
    - `researcher.py` - End-to-end research workflows and experiment coordination
  - **`utils/`** - General utilities and model handling
    - `datasets.py` - Dataset loading utilities for experiments
    - Model architecture utilities for different LLM types (Phi-3, Falcon, MoE models)

### Data Organization
- **`data/`** - Experimental data and results
- **`notebooks/`** - Jupyter notebooks for exploration and visualization
- **`scripts/`** - Executable scripts for running experiments

### Key Base Classes & Files
- **`SuperWeight` dataclass** - Core data structure representing detected super weights
- **`Detector`** - Pure detection logic for identifying super weights across architectures
- **`Manager`** - Context managers for safe weight manipulation experiments
- **`Analyzer`** - Main coordinator for orchestrating different analysis types
- **`Researcher`** - High-level workflows for complete research experiments

## Key Principles
- **Separation of concerns**: Detection, management, analysis, and research are distinct modules
- **Architecture agnostic**: Support multiple LLM architectures (GPT, BERT, Phi-3, Falcon, MoE)
- **Safe experimentation**: Use context managers for weight modifications to prevent corruption
- **Linguistic focus**: Analyze super weights' role in language processing and representation
- **Reproducible research**: Clear workflows from detection through analysis to results

## Research Focus Areas
Based on the project description, maintain focus on:
1. **Extension to new architectures** - Phi-3, Falcon, MoE models
2. **Functional analysis** - Linguistic phenomena, connectivity patterns, massive activations
3. **Manipulation experiments** - Controlled perturbations and capability enhancement
4. **Relationship analysis** - Super weights ↔ massive activations ↔ privileged basis

## Python Guidelines
- Use type hints for all function signatures: `dict`, `list`, `tuple`, `| None`
- Use dataclasses for structured data (like `SuperWeight`)
- Import at the top of files, organize by standard library, third-party, local imports
- Use descriptive variable names: `is_super_weight`, `has_massive_activation`, `detected_weights`
- Use lowercase with underscores for files and directories

## Logging
- Use Python's standard `logging` module
- Create logger at module level: `logger = logging.getLogger(__name__)`
- Log levels:
  - `logger.info()` for detection progress and experiment milestones
  - `logger.warning()` for suspicious weight patterns or potential issues
  - `logger.error()` for detection failures and exceptions
  - `logger.debug()` for detailed weight analysis information

## Architecture-Specific Guidelines
- **Model utilities**: Place architecture-specific code in `utils/` with clear naming
- **Detection compatibility**: Ensure `detector.py` works across GPT, BERT, Phi-3, Falcon, MoE
- **Analysis adaptation**: Adapt `vocabulary.py` methods for different tokenization schemes
- **Pattern recognition**: Design `patterns.py` to identify architecture-specific super weight characteristics

## Experimental Workflow
1. **Detection**: Use `Detector` to identify super weights in target models
2. **Analysis**: Coordinate analysis through `Analyzer` using vocabulary, metrics, patterns modules
3. **Manipulation**: Use `Manager` context managers for safe perturbation experiments
4. **Research**: Orchestrate complete workflows through `Researcher` for reproducible experiments

## Data Handling
- **Detection results**: Save SuperWeight objects and detection metadata
- **Analysis outputs**: Store linguistic analysis, connectivity patterns, massive activation data
- **Experimental results**: Document manipulation experiments with before/after metrics
- **Visualization data**: Generate plots for weight distributions, activation patterns, linguistic correlates

## File Organization Best Practices
- **Detection logic**: Keep pure detection algorithms in `detection/detector.py`
- **Weight modifications**: All manipulation code goes through `management/manager.py`
- **Analysis coordination**: Use `analysis/analyzer.py` as main entry point for analysis workflows
- **Research workflows**: High-level experiment coordination in `research/researcher.py`
- **Model-specific utilities**: Architecture handling code in `utils/`
- **Exploration**: Use `notebooks/` for interactive analysis and visualization

## Super Weights Specific Guidelines
- **Detection consistency**: Use standardized methods across different model architectures
- **Linguistic analysis**: Focus on how super weights influence syntactic/semantic processing
- **Massive activations**: Track the relationship between super weights and activation magnitude
- **MoE analysis**: Special attention to expert networks and routing mechanisms
- **Privileged basis**: Investigate connections to "no privileged basis" theoretical assumptions
- **Multilingual analysis**: Test super weight consistency across languages

## Safety and Reproducibility
- **Context managers**: Always use `Manager` for weight modifications to ensure restoration
- **Experiment tracking**: Document all manipulation experiments with precise parameters
- **Model state preservation**: Never permanently modify model weights without explicit intent
- **Reproducible seeds**: Use consistent random seeds for detection and analysis
- **Version control**: Track model versions and analysis parameters for reproducibility

## Debugging
- Use print statements with descriptive variable names: `print(f"{detected_super_weights=}")`
- Log detection progress and analysis milestones prominently
- Use `import code; code.interact(local=dict(globals(), **locals()))` for interactive debugging
- Monitor weight modification experiments carefully with before/after validation