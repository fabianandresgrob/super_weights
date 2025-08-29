import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import List, Literal, Dict, Optional
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import os

from utils.model_architectures import UniversalLayerHandler

class HeadAnalyzer:
    """
    Analyzes attention heads in a specific layer to find the most promising
    targets for attention-based adversarial attacks (Hypotheses D and E).
    """
    def __init__(self, model: torch.nn.Module, tokenizer, layer_handler: UniversalLayerHandler):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_handler = layer_handler
        # Robust device resolution
        self.device = next(model.parameters()).device
        self.model.eval()

    def analyze(
        self,
        layer_idx: int,
        prompts: List[str],
        metric: Literal['sink_strength', 'key_bias_alignment']
    ) -> Dict[int, float]:
        """
        Analyzes all heads in a given layer across multiple prompts.

        Args:
            layer_idx: The index of the layer to analyze.
            prompts: A list of strings to use for the analysis.
            metric: The metric to calculate ('sink_strength' for Hypo E, 
                    'key_bias_alignment' for Hypo D).

        Returns:
            A dictionary mapping head index to its average vulnerability score.
        """
        num_heads = self.layer_handler.get_attention_architecture(layer_idx).num_attention_heads
        head_scores = {i: 0.0 for i in range(num_heads)}
        prompts_processed = 0

        pbar = tqdm(prompts, desc=f"Analyzing Heads in Layer {layer_idx} for {metric}")
        for prompt in pbar:
            try:
                # Set add_special_tokens=True to include the BOS token, which is
                # the actual attention sink we want to analyze
                tokens = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=True).to(self.device)
                
                # Ensure we have at least 2 tokens (BOS + one content token) for analysis
                if tokens.input_ids.shape[1] < 2:
                    continue

                # Request attentions and (when needed) hidden states
                need_hidden = (metric == 'key_bias_alignment')
                with torch.no_grad():
                    outputs = self.model(**tokens, output_attentions=True, output_hidden_states=need_hidden)
                
                attn_probs = outputs.attentions[layer_idx]  # [batch, heads, seq, seq]
                
                if metric == 'sink_strength':
                    scores = self._calculate_sink_strength(attn_probs)
                elif metric == 'key_bias_alignment':
                    hidden_states = outputs.hidden_states[layer_idx]  # requires output_hidden_states=True
                    scores = self._calculate_key_bias_alignment(hidden_states, layer_idx)
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                for i in range(num_heads):
                    head_scores[i] += float(scores[i].item())
                prompts_processed += 1

            except Exception as e:
                print(f"Skipping prompt due to error: {e}")
                # print traceback
                print(traceback.format_exc())
                continue
        
        if prompts_processed == 0:
            raise RuntimeError("Could not process any prompts for analysis.")

        for i in range(num_heads):
            head_scores[i] /= prompts_processed
            
        return head_scores

    def _calculate_sink_strength(self, attn_probs: torch.Tensor) -> torch.Tensor:
        """Calculates attention paid to the first token by subsequent tokens for each head."""
        # attn_probs shape: [batch, heads, seq_len, seq_len]
        # We want sum of probs[:, :, 1:, 0]
        sink_attention = attn_probs[:, :, 1:, 0].sum(dim=-1)
        return sink_attention.mean(dim=0) # Average over batch if batch_size > 1

    def _calculate_key_bias_alignment(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Calculates cosine similarity between k_0 and q_1 for each head."""
        # Get attention architecture info from layer handler
        attn_info = self.layer_handler.get_attention_architecture(layer_idx)
        q_proj = self.layer_handler.get_attention_components(layer_idx)['q_proj']
        k_proj = self.layer_handler.get_attention_components(layer_idx)['k_proj']
        
        norm_comps = self.layer_handler.get_normalization_components(layer_idx)
        if 'input_layernorm' in norm_comps:
            hidden_states = norm_comps['input_layernorm'](hidden_states)

        queries = q_proj(hidden_states)
        keys = k_proj(hidden_states)

        batch_size, seq_len, _ = queries.shape
        if seq_len < 2:
            return torch.zeros(attn_info.num_attention_heads, device=self.device)

        # Use values from attention architecture info
        num_heads = attn_info.num_attention_heads  # 32
        head_dim = attn_info.head_dim  # 128
        num_key_value_heads = getattr(attn_info, 'num_key_value_heads', num_heads)  # 8
        
        # Reshape queries: [batch, seq, num_heads * head_dim] -> [batch, seq, num_heads, head_dim]
        queries = queries.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Reshape keys: [batch, seq, num_key_value_heads * head_dim] -> [batch, seq, num_key_value_heads, head_dim]
        keys = keys.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
        
        # For GQA, we need to repeat keys to match the number of query heads
        # Each key head serves num_heads // num_key_value_heads query heads
        if num_key_value_heads != num_heads:
            # Repeat keys to match query heads: [batch, num_key_value_heads, seq, head_dim] -> [batch, num_heads, seq, head_dim]
            keys = keys.repeat_interleave(num_heads // num_key_value_heads, dim=1)
        
        # k_0: Key of the first token for all heads
        k0 = keys[:, :, 0, :]  # [batch, num_heads, head_dim]
        # q_1: Query of the second token for all heads
        q1 = queries[:, :, 1, :]  # [batch, num_heads, head_dim]
        
        alignment = F.cosine_similarity(k0, q1, dim=-1)
        return alignment.mean(dim=0)  # Average over batch

    @staticmethod
    def plot_head_scores(
        head_scores: Dict[int, float],
        title: str,
        save_path: Optional[str] = None
    ):
        """
        Creates and optionally saves a bar plot of head scores, sorted by score.
        """
        if not head_scores:
            print("Warning: No head scores provided to plot.")
            return

        scores_series = pd.Series(head_scores).sort_values(ascending=False)

        plt.figure(figsize=(16, 7))
        scores_series.plot(kind='bar', color='skyblue', edgecolor='black')
        
        plt.xlabel("Head Index (Sorted by Score)")
        plt.ylabel("Vulnerability Score")
        plt.title(title, fontsize=16)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if len(scores_series) > 40:
            plt.xticks(rotation=90, fontsize=8)
        else:
            plt.xticks(rotation=45)
        
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"Analysis plot saved to: {save_path}")
        
        plt.show()