from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset
import random


class DatasetLoader:
    """
    Utility class for loading and managing datasets for super weight analysis.
    Provides standardized interfaces for common evaluation datasets.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
    
    def load_perplexity_dataset(self, 
                              dataset_name: str = 'wikitext',
                              config: str = 'wikitext-2-raw-v1',
                              split: str = 'test',
                              n_samples: int = 100,
                              min_length: int = 50) -> List[str]:
        """
        Load dataset for perplexity evaluation.
        
        Args:
            dataset_name: Name of the dataset
            config: Dataset configuration
            split: Dataset split
            n_samples: Number of samples to load
            min_length: Minimum text length to include
            
        Returns:
            List of text strings
        """
        
        dataset = load_dataset(dataset_name, config, split=split, streaming=True)
        texts = []
        
        for i, example in enumerate(dataset):
            if i >= n_samples * 3:  # Load extra to account for filtering
                break
            
            text = example['text'].strip()
            if len(text) >= min_length:
                texts.append(text)
            
            if len(texts) >= n_samples:
                break
        
        # If we don't have enough, repeat shorter texts
        if len(texts) < n_samples:
            while len(texts) < n_samples:
                texts.extend(texts[:min(len(texts), n_samples - len(texts))])
        
        return texts[:n_samples]
    
    def load_hellaswag(self, split: str = 'validation', n_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Load HellaSwag dataset for completion tasks.
        
        Args:
            split: Dataset split
            n_samples: Number of samples to load
            
        Returns:
            List of HellaSwag examples
        """
        
        dataset = load_dataset('hellaswag', split=split, streaming=True)
        examples = []
        
        for i, example in enumerate(dataset):
            if i >= n_samples:
                break
            
            examples.append({
                'context': example['ctx'],
                'endings': example['endings'],
                'label': int(example['label']),
                'activity_label': example.get('activity_label', ''),
                'source_id': example.get('source_id', '')
            })
        
        return examples
    
    def load_arc(self, difficulty: str = 'easy', 
                 split: str = 'test', 
                 n_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Load ARC (AI2 Reasoning Challenge) dataset.
        
        Args:
            difficulty: 'easy' or 'challenge'
            split: Dataset split
            n_samples: Number of samples to load
            
        Returns:
            List of ARC examples
        """
        
        config = 'ARC-Easy' if difficulty == 'easy' else 'ARC-Challenge'
        dataset = load_dataset('ai2_arc', config, split=split, streaming=True)
        examples = []
        
        for i, example in enumerate(dataset):
            if i >= n_samples:
                break
            
            # Find the correct answer index
            answer_key = example['answerKey']
            choices = example['choices']
            
            try:
                label = choices['label'].index(answer_key)
            except ValueError:
                # Skip examples where answer key doesn't match choices
                continue
            
            examples.append({
                'question': example['question'],
                'choices': choices['text'],
                'label': label,
                'answer_key': answer_key,
                'id': example.get('id', '')
            })
        
        return examples
    
    def load_truthfulqa(self, split: str = 'validation', n_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Load TruthfulQA dataset.
        
        Args:
            split: Dataset split
            n_samples: Number of samples to load
            
        Returns:
            List of TruthfulQA examples
        """
        
        dataset = load_dataset('truthful_qa', 'multiple_choice', split=split, streaming=True)
        examples = []
        
        for i, example in enumerate(dataset):
            if i >= n_samples:
                break
            
            # Find correct answer
            correct_answers = [i for i, label in enumerate(example['mc2_targets']['labels']) if label == 1]
            
            if correct_answers:
                examples.append({
                    'question': example['question'],
                    'choices': example['mc2_targets']['choices'],
                    'labels': example['mc2_targets']['labels'],
                    'correct_answers': correct_answers,
                    'category': example.get('category', ''),
                    'source': example.get('source', '')
                })
        
        return examples
    
    def load_mmlu(self, subject: str = 'abstract_algebra',
                  split: str = 'test',
                  n_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Load MMLU (Massive Multitask Language Understanding) dataset.
        
        Args:
            subject: Subject area
            split: Dataset split
            n_samples: Number of samples to load
            
        Returns:
            List of MMLU examples
        """
        
        dataset = load_dataset('cais/mmlu', subject, split=split, streaming=True)
        examples = []
        
        for i, example in enumerate(dataset):
            if i >= n_samples:
                break
            
            examples.append({
                'question': example['question'],
                'choices': [example['A'], example['B'], example['C'], example['D']],
                'label': example['answer'],  # 0, 1, 2, or 3
                'subject': subject
            })
        
        return examples
    
    def load_custom_text_samples(self, 
                                domain: str = 'general',
                                n_samples: int = 50) -> List[str]:
        """
        Load custom text samples for specific domains.
        
        Args:
            domain: Domain type ('general', 'code', 'math', 'scientific', 'conversational')
            n_samples: Number of samples to generate
            
        Returns:
            List of text strings
        """
        
        domain_templates = {
            'general': [
                "The researchers are investigating new approaches to {topic}.",
                "Recent developments in {field} have shown promising results.",
                "According to the latest studies, {claim} appears to be true.",
                "The implementation of {technology} has revolutionized {industry}.",
                "Experts believe that {prediction} will occur within the next decade."
            ],
            'code': [
                "def {function_name}({params}): return {expression}",
                "class {class_name}: def __init__(self): self.{attribute} = {value}",
                "for {var} in {iterable}: if {condition}: {action}",
                "import {module}; {module}.{function}({args})",
                "try: {code_block} except {exception}: {handler}"
            ],
            'math': [
                "The derivative of {function} is {derivative}.",
                "Given that {equation}, we can solve for {variable}.",
                "The integral of {function} from {a} to {b} equals {result}.",
                "In the triangle with sides {a}, {b}, {c}, the area is {area}.",
                "The probability of {event} occurring is {probability}."
            ],
            'scientific': [
                "The experiment demonstrated that {hypothesis} under {conditions}.",
                "Analysis of {data_type} revealed {finding} with significance p < {p_value}.",
                "The {species} exhibits {behavior} during {time_period}.",
                "Measurements indicate that {parameter} varies by {amount} across {samples}.",
                "The reaction between {reactant1} and {reactant2} produces {product}."
            ],
            'conversational': [
                "How are you doing today? I hope everything is going well.",
                "Did you see the news about {topic}? It's quite interesting.",
                "I was thinking about {subject} and wondered what you think.",
                "Thanks for helping me with {task}. I really appreciate it.",
                "Could you please explain {concept} in simpler terms?"
            ]
        }
        
        templates = domain_templates.get(domain, domain_templates['general'])
        samples = []
        
        # Generate samples by filling templates
        placeholders = {
            'topic': ['machine learning', 'climate change', 'renewable energy', 'artificial intelligence'],
            'field': ['technology', 'medicine', 'education', 'transportation'],
            'claim': ['this approach is effective', 'the results are significant', 'the method works'],
            'technology': ['blockchain', 'quantum computing', 'neural networks', 'robotics'],
            'industry': ['healthcare', 'finance', 'manufacturing', 'agriculture'],
            'prediction': ['automation will increase', 'costs will decrease', 'efficiency will improve'],
            'function_name': ['calculate', 'process', 'analyze', 'transform'],
            'params': ['x, y', 'data', 'values', 'items'],
            'expression': ['x + y', 'len(data)', 'max(values)', 'sum(items)'],
            'class_name': ['DataProcessor', 'Calculator', 'Analyzer', 'Handler'],
            'attribute': ['data', 'value', 'result', 'status'],
            'value': ['None', '0', '[]', 'True'],
            'var': ['item', 'x', 'element', 'value'],
            'iterable': ['items', 'data', 'range(10)', 'values'],
            'condition': ['x > 0', 'item is not None', 'len(data) > 0'],
            'action': ['print(item)', 'process(x)', 'continue'],
            'module': ['numpy', 'pandas', 'matplotlib', 'torch'],
            'function': ['array', 'DataFrame', 'plot', 'tensor'],
            'args': ['data', '[1, 2, 3]', 'x=5', 'shape=(10,)'],
            'code_block': ['result = compute()', 'data = load_file()', 'model.train()'],
            'exception': ['ValueError', 'FileNotFoundError', 'KeyError'],
            'handler': ['print("Error")', 'return None', 'pass'],
            'function': ['x^2', 'sin(x)', 'log(x)', 'e^x'],
            'derivative': ['2x', 'cos(x)', '1/x', 'e^x'],
            'equation': ['x^2 + y^2 = 1', 'y = mx + b', 'f(x) = ax + b'],
            'variable': ['x', 'y', 'a', 'b'],
            'a': ['0', '1', '-1', 'π'],
            'b': ['1', '2', 'π', '∞'],
            'result': ['π/2', '0', '1', 'e'],
            'area': ['1/2 * base * height', 'π * r^2', 'side^2'],
            'event': ['success', 'failure', 'occurrence'],
            'probability': ['0.5', '0.1', '0.95', '1/6'],
            'hypothesis': ['the treatment is effective', 'the correlation exists'],
            'conditions': ['controlled temperature', 'standard pressure', 'laboratory conditions'],
            'data_type': ['genomic data', 'survey responses', 'sensor readings'],
            'finding': ['significant correlation', 'unexpected pattern', 'clear trend'],
            'p_value': ['0.05', '0.01', '0.001'],
            'species': ['dolphins', 'honeybees', 'migratory birds'],
            'behavior': ['cooperative hunting', 'complex communication', 'seasonal migration'],
            'time_period': ['breeding season', 'winter months', 'mating period'],
            'parameter': ['temperature', 'pH level', 'concentration'],
            'amount': ['±2°C', '0.5 units', '10%'],
            'samples': ['different locations', 'various conditions', 'multiple trials'],
            'reactant1': ['hydrogen', 'sodium', 'carbon dioxide'],
            'reactant2': ['oxygen', 'chlorine', 'water'],
            'product': ['water', 'salt', 'carbonic acid'],
            'subject': ['the weather', 'your project', 'that movie'],
            'task': ['the presentation', 'moving furniture', 'coding'],
            'concept': ['quantum mechanics', 'machine learning', 'economics']
        }
        
        for i in range(n_samples):
            template = random.choice(templates)
            
            # Fill in placeholders
            filled_template = template
            for placeholder, options in placeholders.items():
                if '{' + placeholder + '}' in filled_template:
                    filled_template = filled_template.replace(
                        '{' + placeholder + '}', 
                        random.choice(options)
                    )
            
            samples.append(filled_template)
        
        return samples
    
    def get_standard_test_suite(self) -> Dict[str, List]:
        """
        Get a standard test suite with samples from multiple domains.
        
        Returns:
            Dictionary with test samples organized by domain
        """
        
        return {
            'perplexity_texts': self.load_perplexity_dataset(n_samples=50),
            'hellaswag': self.load_hellaswag(n_samples=50),
            'arc_easy': self.load_arc(difficulty='easy', n_samples=50),
            'general_texts': self.load_custom_text_samples('general', 25),
            'code_texts': self.load_custom_text_samples('code', 25),
            'math_texts': self.load_custom_text_samples('math', 25),
            'scientific_texts': self.load_custom_text_samples('scientific', 25),
            'conversational_texts': self.load_custom_text_samples('conversational', 25)
        }
    
    def create_multilingual_test_set(self, languages: List[str] = None) -> Dict[str, List[str]]:
        """
        Create a test set with simple sentences in multiple languages.
        
        Args:
            languages: List of language codes
            
        Returns:
            Dictionary mapping language codes to text samples
        """
        
        if languages is None:
            languages = ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ar', 'hi']
        
        # Simple test sentences (these would ideally be loaded from proper multilingual datasets)
        base_sentences = {
            'en': [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is transforming technology.",
                "The weather is beautiful today.",
                "I enjoy reading books in my free time.",
                "Science and technology advance rapidly."
            ],
            'es': [
                "El rápido zorro marrón salta sobre el perro perezoso.",
                "El aprendizaje automático está transformando la tecnología.",
                "El clima está hermoso hoy.",
                "Disfruto leyendo libros en mi tiempo libre.",
                "La ciencia y la tecnología avanzan rápidamente."
            ],
            'fr': [
                "Le renard brun rapide saute par-dessus le chien paresseux.",
                "L'apprentissage automatique transforme la technologie.",
                "Le temps est magnifique aujourd'hui.",
                "J'aime lire des livres pendant mon temps libre.",
                "La science et la technologie progressent rapidement."
            ],
            'de': [
                "Der schnelle braune Fuchs springt über den faulen Hund.",
                "Maschinelles Lernen verändert die Technologie.",
                "Das Wetter ist heute schön.",
                "Ich lese gerne Bücher in meiner Freizeit.",
                "Wissenschaft und Technologie entwickeln sich schnell."
            ],
            'zh': [
                "快速的棕色狐狸跳过懒狗。",
                "机器学习正在改变技术。",
                "今天天气很好。",
                "我喜欢在空闲时间读书。",
                "科学技术发展迅速。"
            ],
            'ja': [
                "素早い茶色のキツネが怠惰な犬を飛び越える。",
                "機械学習が技術を変革している。",
                "今日は天気が美しい。",
                "自由時間に本を読むのが好きです。",
                "科学技術は急速に進歩している。"
            ],
            'ar': [
                "الثعلب البني السريع يقفز فوق الكلب الكسول.",
                "التعلم الآلي يغير التكنولوجيا.",
                "الطقس جميل اليوم.",
                "أستمتع بقراءة الكتب في وقت فراغي.",
                "العلم والتكنولوجيا يتقدمان بسرعة."
            ],
            'hi': [
                "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूदती है।",
                "मशीन लर्निंग तकनीक को बदल रही है।",
                "आज मौसम सुंदर है।",
                "मुझे अपने खाली समय में किताबें पढ़ना पसंद है।",
                "विज्ञान और तकनीक तेजी से आगे बढ़ रहे हैं।"
            ]
        }
        
        multilingual_set = {}
        for lang in languages:
            if lang in base_sentences:
                multilingual_set[lang] = base_sentences[lang]
            else:
                # Fallback to English if language not available
                multilingual_set[lang] = base_sentences['en']
        
        return multilingual_set