import torch
import numpy as np
import re
from typing import List, Dict, Any, Optional
from datasets import load_dataset

from detection.super_weight import SuperWeight
from utils.model_architectures import UniversalMLPHandler


class MetricsAnalyzer:
    """
    Analyzes performance metrics (perplexity, accuracy) with super weight modifications.
    """
    
    def __init__(self, model, tokenizer, manager, mlp_handler: UniversalMLPHandler):
        self.model = model
        self.tokenizer = tokenizer
        self.manager = manager
        self.mlp_handler = mlp_handler  # Use passed handler instead of creating new one
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def measure_perplexity_impact(self, super_weight: SuperWeight | List[SuperWeight], 
                            dataset_name: str = 'wikitext',
                            dataset_config: str = 'wikitext-2-raw-v1',
                            split: str = 'test',
                            n_samples: int = 100,
                            max_length: int = 512) -> Dict[str, Any]:
        """
        Measure perplexity impact of a super weight or list of super weights using causal intervention.
        
        Args:
            super_weight: SuperWeight or List[SuperWeight] to analyze
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration
            split: Dataset split to use
            n_samples: Number of samples to evaluate
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with perplexity metrics
        """
        
        # Handle both single SuperWeight and list of SuperWeights
        if isinstance(super_weight, list):
            super_weights = super_weight
            is_multiple = True
        else:
            super_weights = [super_weight]
            is_multiple = False
        
        # Load dataset
        dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=False)
        dataset = dataset.shuffle(seed=42)  # Shuffle for randomness
        texts = []
        
        # Sample texts
        for i, example in enumerate(dataset):
            if i >= n_samples:
                break
            text = example['text'].strip()
            if len(text) > 50:  # Skip very short texts
                texts.append(text)
        
        if len(texts) < n_samples:
            texts = texts * ((n_samples // len(texts)) + 1)
            texts = texts[:n_samples]
        
        # Measure baseline perplexity
        baseline_perplexity = self._compute_perplexity(texts, max_length)
        
        # Measure perplexity with super weight(s) zeroed
        with self.manager.temporary_zero(super_weights):
            modified_perplexity = self._compute_perplexity(texts, max_length)
        
        # Calculate impact metrics
        perplexity_ratio = modified_perplexity / baseline_perplexity
        perplexity_increase = modified_perplexity - baseline_perplexity
        
        # Build result dictionary based on whether it's single or multiple
        result = {
            'baseline_perplexity': baseline_perplexity,
            'modified_perplexity': modified_perplexity,
            'perplexity_ratio': perplexity_ratio,
            'perplexity_increase': perplexity_increase,
            'impact_severity': self._classify_perplexity_impact(perplexity_ratio),
            'dataset_info': {
                'name': dataset_name,
                'config': dataset_config,
                'split': split,
                'n_samples': len(texts)
            }
        }
        
        if is_multiple:
            result.update({
                'super_weights': super_weights,
                'num_weights': len(super_weights),
                'average_impact_per_weight': perplexity_increase / len(super_weights) if super_weights else 0.0,
            })
        else:
            result['super_weight'] = super_weight
        
        return result
    
    def _compute_perplexity(self, texts: List[str], max_length: int) -> float:
        """Compute perplexity on a list of texts"""
        total_loss = 0.0
        total_tokens = 0
        
        for text in texts:
            # Tokenize
            encoding = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.model.device)
            
            # Skip very short sequences
            if input_ids.shape[1] < 2:
                continue
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss
                
                # Accumulate loss and token count
                total_loss += loss.item() * input_ids.numel()
                total_tokens += input_ids.numel()
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def _classify_perplexity_impact(self, perplexity_ratio: float) -> str:
        """Classify the severity of perplexity impact"""
        if perplexity_ratio > 10.0:
            return "catastrophic"
        elif perplexity_ratio > 3.0:
            return "severe"
        elif perplexity_ratio > 1.5:
            return "moderate"
        elif perplexity_ratio > 1.1:
            return "mild"
        else:
            return "minimal"
    
    def measure_accuracy_impact(self, super_weight: SuperWeight | List[SuperWeight],
                          task: str = 'hellaswag',
                          n_samples: int = 100) -> Dict[str, Any]:
        """
        Measure accuracy impact on downstream tasks.
        
        Args:
            super_weight: SuperWeight or List[SuperWeight] to analyze
            task: Task name (hellaswag, arc_easy, arc_challenge, etc.)
            n_samples: Number of samples to evaluate
            
        Returns:
            Dictionary with accuracy metrics
        """
        
        # Handle both single SuperWeight and list of SuperWeights
        if isinstance(super_weight, list):
            super_weights = super_weight
            is_multiple = True
        else:
            super_weights = [super_weight]
            is_multiple = False
    
        # Load task data
        task_data = self._load_task_data(task, n_samples)
        
        if not task_data:
            result = {
                'error': f"Could not load task data for {task}",
                'task': task
            }
            if is_multiple:
                result.update({
                    'super_weights': super_weights,
                    'num_weights': len(super_weights)
                })
            else:
                result['super_weight'] = super_weight
            return result
        
        # Measure baseline accuracy
        baseline_accuracy = self._compute_accuracy(task_data, task)
        
        # Measure accuracy with super weight(s) zeroed
        with self.manager.temporary_zero(super_weights):
            modified_accuracy = self._compute_accuracy(task_data, task)
        
        # Calculate impact metrics
        accuracy_drop = baseline_accuracy - modified_accuracy
        accuracy_ratio = modified_accuracy / baseline_accuracy if baseline_accuracy > 0 else 0.0
        
        result = {
            'task': task,
            'baseline_accuracy': baseline_accuracy,
            'modified_accuracy': modified_accuracy,
            'accuracy_drop': accuracy_drop,
            'accuracy_ratio': accuracy_ratio,
            'impact_severity': self._classify_accuracy_impact(accuracy_drop),
            'n_samples': len(task_data)
        }
        
        if is_multiple:
            result.update({
                'super_weights': super_weights,
                'num_weights': len(super_weights),
                'average_impact_per_weight': accuracy_drop / len(super_weights) if super_weights else 0.0,
            })
        else:
            result['super_weight'] = super_weight
    
        return result
    
    def _load_task_data(self, task: str, n_samples: int) -> Optional[List[Dict]]:
        """Load data for a specific task"""
        try:
            if task == 'hellaswag':
                dataset = load_dataset('hellaswag', split='validation', streaming=False)
                dataset = dataset.shuffle(seed=42)  # Shuffle for randomness
                data = []
                for i, example in enumerate(dataset):
                    if i >= n_samples:
                        break
                    data.append({
                        'context': example['ctx'],
                        'endings': example['endings'],
                        'label': int(example['label'])
                    })
                return data
            
            elif task == 'arc_easy':
                dataset = load_dataset('ai2_arc', 'ARC-Easy', split='test', streaming=False)
                data = []
                for i, example in enumerate(dataset):
                    if i >= n_samples:
                        break
                    data.append({
                        'question': example['question'],
                        'choices': example['choices']['text'],
                        'label': example['choices']['label'].index(example['answerKey'])
                    })
                return data
            
            elif task == 'arc_challenge':
                dataset = load_dataset('ai2_arc', 'ARC-Challenge', split='test', streaming=False)
                data = []
                for i, example in enumerate(dataset):
                    if i >= n_samples:
                        break
                    data.append({
                        'question': example['question'],
                        'choices': example['choices']['text'],
                        'label': example['choices']['label'].index(example['answerKey'])
                    })
                return data
            
            elif task == 'mmlu':
                # Load multiple MMLU subjects
                subjects = ['abstract_algebra', 'anatomy', 'business_ethics']
                all_data = []
                samples_per_subject = max(1, n_samples // len(subjects))
                
                for subject in subjects:
                    dataset = load_dataset('cais/mmlu', subject, split='test', streaming=False)
                    subject_data = []
                    for i, example in enumerate(dataset):
                        if i >= samples_per_subject:
                            break
                        subject_data.append({
                            'question': example['question'],
                            'choices': [example['A'], example['B'], example['C'], example['D']],
                            'label': example['answer'],  # 0, 1, 2, or 3
                            'subject': subject
                        })
                    all_data.extend(subject_data)
                
                return all_data[:n_samples]
            
            elif task == 'gsm8k':
                dataset = load_dataset('gsm8k', 'main', split='test', streaming=False)
                data = []
                for i, example in enumerate(dataset):
                    if i >= n_samples:
                        break
                    
                    # Extract numerical answer from the solution
                    answer_text = example['answer']
                    # Find the final numerical answer (typically after #### in GSM8K)
                    if '####' in answer_text:
                        answer_str = answer_text.split('####')[-1].strip()
                    else:
                        # Try to extract the last number in the answer
                        numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_text)
                        answer_str = numbers[-1] if numbers else "0"
                    
                    try:
                        answer_num = float(answer_str.replace(',', ''))
                    except:
                        answer_num = 0.0
                    
                    data.append({
                        'question': example['question'],
                        'answer': answer_num,
                        'answer_text': answer_text
                    })
                return data
            
            else:
                return None
                
        except Exception as e:
            print(f"Error loading task {task}: {e}")
            return None
    
    def _compute_accuracy(self, task_data: List[Dict], task: str) -> float:
        """Compute accuracy on task data"""
        correct = 0
        total = 0
        
        for example in task_data:
            try:
                if task == 'hellaswag':
                    prediction = self._predict_hellaswag(example)
                elif task in ['arc_easy', 'arc_challenge']:
                    prediction = self._predict_arc(example)
                elif task == 'mmlu':
                    prediction = self._predict_mmlu(example)
                elif task == 'gsm8k':
                    prediction = self._predict_gsm8k(example)
                else:
                    continue
                
                if task == 'gsm8k':
                    # For GSM8K, check if the predicted answer is close to the correct one
                    if abs(prediction - example['answer']) < 0.01:
                        correct += 1
                else:
                    if prediction == example['label']:
                        correct += 1
                total += 1
                
            except Exception as e:
                # Skip examples that cause errors
                continue
        
        return correct / total if total > 0 else 0.0
    
    def _predict_hellaswag(self, example: Dict) -> int:
        """Predict HellaSwag completion"""
        context = example['context']
        endings = example['endings']
        
        best_score = float('-inf')
        best_idx = 0
        
        for i, ending in enumerate(endings):
            # Create full text
            full_text = context + " " + ending
            
            # Tokenize
            tokens = self.tokenizer(full_text, return_tensors='pt').to(self.model.device)
            
            # Compute log probability
            with torch.no_grad():
                outputs = self.model(**tokens, labels=tokens['input_ids'])
                log_prob = -outputs.loss.item()
                
                if log_prob > best_score:
                    best_score = log_prob
                    best_idx = i
        
        return best_idx
    
    def _predict_arc(self, example: Dict) -> int:
        """Predict ARC answer"""
        question = example['question']
        choices = example['choices']
        
        best_score = float('-inf')
        best_idx = 0
        
        for i, choice in enumerate(choices):
            # Create question-answer pair
            qa_text = f"Question: {question}\nAnswer: {choice}"
            
            # Tokenize
            tokens = self.tokenizer(qa_text, return_tensors='pt').to(self.model.device)
            
            # Compute log probability
            with torch.no_grad():
                outputs = self.model(**tokens, labels=tokens['input_ids'])
                log_prob = -outputs.loss.item()
                
                if log_prob > best_score:
                    best_score = log_prob
                    best_idx = i
        
        return best_idx
    
    def _predict_mmlu(self, example: Dict) -> int:
        """Predict MMLU answer"""
        question = example['question']
        choices = example['choices']
        
        best_score = float('-inf')
        best_idx = 0
        
        for i, choice in enumerate(choices):
            # Create question-answer format
            qa_text = f"Question: {question}\n\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\nAnswer: {choice}"
            
            # Tokenize
            tokens = self.tokenizer(qa_text, return_tensors='pt', truncation=True, max_length=512).to(self.model.device)
            
            # Compute log probability
            with torch.no_grad():
                outputs = self.model(**tokens, labels=tokens['input_ids'])
                log_prob = -outputs.loss.item()
                
                if log_prob > best_score:
                    best_score = log_prob
                    best_idx = i
        
        return best_idx
    
    def _predict_gsm8k(self, example: Dict) -> float:
        """Predict GSM8K numerical answer"""
        question = example['question']
        
        # Create a prompt for numerical reasoning
        prompt = f"Question: {question}\nLet's think step by step.\nAnswer:"
        
        # Tokenize
        tokens = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(self.model.device)
        
        # Generate answer
        with torch.no_grad():
            output = self.model.generate(
                tokens['input_ids'],
                max_new_tokens=100,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(output[0][tokens['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract numerical answer
        numbers = re.findall(r'-?\d+(?:\.\d+)?', generated_text)
        if numbers:
            try:
                return float(numbers[-1].replace(',', ''))
            except:
                return 0.0
        else:
            return 0.0
    
    def _classify_accuracy_impact(self, accuracy_drop: float) -> str:
        """Classify the severity of accuracy impact"""
        if accuracy_drop > 0.5:
            return "catastrophic"
        elif accuracy_drop > 0.2:
            return "severe"
        elif accuracy_drop > 0.1:
            return "moderate"
        elif accuracy_drop > 0.05:
            return "mild"
        else:
            return "minimal"
    
    def comprehensive_impact_analysis(self, super_weight: SuperWeight,
                                    perplexity_config: Dict = None,
                                    accuracy_tasks: List[str] = None) -> Dict[str, Any]:
        """
        Run analysis on multiple metrics.
        
        Args:
            super_weight: SuperWeight to analyze
            perplexity_config: Configuration for perplexity measurement
            accuracy_tasks: List of tasks for accuracy measurement
            
        Returns:
            Dictionary with results across all metrics
        """
        
        if perplexity_config is None:
            perplexity_config = {'n_samples': 50}
        
        if accuracy_tasks is None:
            accuracy_tasks = ['hellaswag']
        
        results = {
            'super_weight': super_weight,
            'perplexity_analysis': self.measure_perplexity_impact(super_weight, **perplexity_config),
            'accuracy_analyses': {}
        }
        
        # Run accuracy analysis for each task
        for task in accuracy_tasks:
            try:
                results['accuracy_analyses'][task] = self.measure_accuracy_impact(super_weight, task, n_samples=50)
            except Exception as e:
                results['accuracy_analyses'][task] = {'error': str(e)}
        
        # Compute overall impact score
        results['overall_impact'] = self._compute_overall_impact_score(results)
        
        return results
    
    def _compute_overall_impact_score(self, results: Dict) -> Dict[str, Any]:
        """Compute an overall impact score combining multiple metrics"""
        
        # Perplexity impact score (0-10)
        perp_ratio = results['perplexity_analysis']['perplexity_ratio']
        perp_score = min(10.0, max(0.0, (perp_ratio - 1.0) * 5.0))
        
        # Accuracy impact score (0-10)
        acc_scores = []
        for task, analysis in results['accuracy_analyses'].items():
            if 'accuracy_drop' in analysis:
                acc_drop = analysis['accuracy_drop']
                acc_score = min(10.0, max(0.0, acc_drop * 20.0))  # Scale to 0-10
                acc_scores.append(acc_score)
        
        avg_acc_score = np.mean(acc_scores) if acc_scores else 0.0
        
        # Combined score (weighted average)
        overall_score = 0.6 * perp_score + 0.4 * avg_acc_score
        
        return {
            'overall_score': overall_score,
            'perplexity_score': perp_score,
            'accuracy_score': avg_acc_score,
            'impact_classification': self._classify_overall_impact(overall_score)
        }
    
    def _classify_overall_impact(self, score: float) -> str:
        """Classify overall impact based on combined score"""
        if score > 8.0:
            return "critical"
        elif score > 6.0:
            return "high"
        elif score > 4.0:
            return "medium"
        elif score > 2.0:
            return "low"
        else:
            return "minimal"
