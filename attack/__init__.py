"""Attack module for super weight research."""

# Public API from attack.attack
from .attack import (
    SuperWeightAttacker,
    MultiPromptSuperWeightAttacker,
    SuperWeightAttackConfig,
    SuperWeightTarget,
    CompositeLoss,
    HypothesisA,
    HypothesisB,
    HypothesisC,
    HypothesisD,
    HypothesisE,
)

from .attack_eval import (
    sample_wikitext_prompts_filtered,
    run_perplexity_bakeoff,
    ppl_for_prompts,
    run_multi_seed_consistency_evaluation
)

__all__ = [
    "SuperWeightAttacker",
    "MultiPromptSuperWeightAttacker",
    "SuperWeightAttackConfig",
    "SuperWeightTarget",
    "CompositeLoss",
    "HypothesisA",
    "HypothesisB",
    "HypothesisC",
    "HypothesisD",
    "HypothesisE",
    "sample_wikitext_prompts_filtered",
    "run_perplexity_bakeoff",
    "ppl_for_prompts",
    "run_multi_seed_consistency_evaluation"
]
