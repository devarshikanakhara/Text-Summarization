"""
Text Summarization using PEGASUS LLM with XSum Dataset
Complete solution for factual and detailed news article summarization
FIXED: Streamlit session state widget modification error
"""

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    pipeline
)
import streamlit as st
import logging
from typing import List, Dict, Any, Optional, Tuple
import warnings
from tqdm import tqdm
import random
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib
import json
import os
import re
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="PEGASUS News Summarizer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1E88E5;
    }
    .success-box {
        background-color: #E8F5E8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4CAF50;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #FF9800;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #1565C0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .model-indicator {
        padding: 0.5rem;
        border-radius: 0.5rem;
        background-color: #E3F2FD;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class XSumDataLoader:
    """
    Handle loading and preprocessing of XSum dataset from Hugging Face
    """
    
    def __init__(self, dataset_name: str = "xsum"):
        """
        Initialize the XSum data loader
        
        Args:
            dataset_name: Name of the dataset on Hugging Face
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self.df = None
        
    def load_dataset(self, split: str = "train", sample_size: Optional[int] = None):
        """
        Load XSum dataset from Hugging Face using load_dataset function
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            sample_size: Number of samples to load (None for all)
            
        Returns:
            Loaded dataset
        """
        try:
            logger.info(f"Loading {self.dataset_name} dataset - {split} split using load_dataset()")
            
            # Using load_dataset function as specified in the hint
            if sample_size:
                # Load full split then select subset
                full_dataset = load_dataset(self.dataset_name, split=split)
                # Randomly sample if sample_size specified
                indices = random.sample(range(len(full_dataset)), min(sample_size, len(full_dataset)))
                self.dataset = full_dataset.select(indices)
                logger.info(f"Loaded {len(self.dataset)} samples (sampled from {len(full_dataset)} total)")
            else:
                self.dataset = load_dataset(self.dataset_name, split=split)
                logger.info(f"Loaded {len(self.dataset)} samples")
            
            # Convert to pandas DataFrame for analysis
            self.df = pd.DataFrame({
                'document': self.dataset['document'],
                'summary': self.dataset['summary'],
                'id': self.dataset['id']
            })
            
            return self.dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            st.error(f"Error loading dataset: {str(e)}")
            raise
    
    def get_statistics(self):
        """Get statistics about the loaded dataset"""
        if self.df is None:
            logger.warning("No dataset loaded. Call load_dataset() first.")
            return {}
        
        stats = {
            'total_samples': len(self.df),
            'avg_document_length': self.df['document'].str.len().mean(),
            'avg_summary_length': self.df['summary'].str.len().mean(),
            'median_document_length': self.df['document'].str.len().median(),
            'median_summary_length': self.df['summary'].str.len().median(),
            'max_document_length': self.df['document'].str.len().max(),
            'min_document_length': self.df['document'].str.len().min(),
            'max_summary_length': self.df['summary'].str.len().max(),
            'min_summary_length': self.df['summary'].str.len().min(),
            'doc_word_count_avg': self.df['document'].str.split().str.len().mean(),
            'summary_word_count_avg': self.df['summary'].str.split().str.len().mean()
        }
        
        return stats
    
    def get_sample(self, index: int = 0):
        """Get a specific sample from the dataset"""
        if self.dataset is None:
            logger.warning("No dataset loaded")
            return None
        
        return {
            'document': self.dataset[index]['document'],
            'summary': self.dataset[index]['summary'],
            'id': self.dataset[index]['id']
        }
    
    def get_random_sample(self):
        """Get a random sample from the dataset"""
        if self.dataset is None:
            return None
        idx = random.randint(0, len(self.dataset) - 1)
        return self.get_sample(idx)
    
    def search_articles(self, keyword: str, max_results: int = 10):
        """Search articles containing a keyword"""
        if self.df is None:
            return []
        
        mask = self.df['document'].str.contains(keyword, case=False, na=False)
        results = self.df[mask].head(max_results)
        return results.to_dict('records')

class PegasusSummarizer:
    """
    Text summarization system using Google's PEGASUS LLM
    PEGASUS is specifically designed for summarization with gap-sentence pre-training
    """
    
    # Model information for display
    MODEL_INFO = {
        "google/pegasus-cnn_dailymail": {
            "name": "PEGASUS CNN/DailyMail",
            "description": "Best for detailed news summarization",
            "characteristics": "Produces 2-3 sentence summaries, excellent factual retention",
            "best_for": "Standard news articles, detailed summaries"
        },
        "google/pegasus-xsum": {
            "name": "PEGASUS XSum",
            "description": "Specialized for extreme summarization",
            "characteristics": "Produces single-sentence summaries, very concise",
            "best_for": "BBC articles, headline-style summaries"
        },
        "google/pegasus-large": {
            "name": "PEGASUS Large",
            "description": "General purpose summarization",
            "characteristics": "Balanced between detail and conciseness",
            "best_for": "Mixed content, general use cases"
        }
    }
    
    def __init__(self, model_name: str = "google/pegasus-cnn_dailymail"):
        """
        Initialize the summarizer with PEGASUS model
        
        Args:
            model_name: HuggingFace model identifier for PEGASUS
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        
    def _load_model(self):
        """Load the PEGASUS model and tokenizer"""
        try:
            model_info = self.MODEL_INFO.get(self.model_name, {})
            with st.spinner(f"Loading {model_info.get('name', self.model_name)}... This may take a few minutes..."):
                logger.info(f"Loading model: {self.model_name}")
                
                # Load PEGASUS tokenizer and model
                self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
                self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name)
                
                # Move model to device
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # Create pipeline for easy inference
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
                
                logger.info(f"PEGASUS model {self.model_name} loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            st.error(f"Error loading model: {str(e)}")
            raise
    
    def clean_summary_text(self, text: str) -> str:
        """
        Clean special tokens and artifacts from generated summary
        
        Args:
            text: Raw summary text from model
            
        Returns:
            Cleaned summary text
        """
        if not text:
            return text
        
        # Replace all HTML-like tokens (<n>, <s>, </s>, <pad>, etc.)
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Fix punctuation spacing
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        # Remove spaces before apostrophes
        text = re.sub(r'\s+\'', '\'', text)
        
        # Fix common artifacts
        text = text.replace(' .', '.').replace(' ,', ',').replace(' ?', '?').replace(' !', '!')
        text = text.replace('( ', '(').replace(' )', ')')
        text = text.replace('[ ', '[').replace(' ]', ']')
        
        # Remove leading/trailing spaces
        text = text.strip()
        
        # Capitalize first letter if needed
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Ensure sentence ends with punctuation
        if text and text[-1] not in ['.', '!', '?']:
            text += '.'
        
        return text
    
    def preprocess_text(self, text: str, max_length: int = 1024) -> str:
        """
        Preprocess input text for summarization
        
        Args:
            text: Input text to preprocess
            max_length: Maximum length for truncation
            
        Returns:
            Preprocessed text
        """
        try:
            # Remove extra whitespaces
            text = " ".join(text.split())
            
            # Basic cleaning
            text = text.replace('\n', ' ').replace('\r', ' ')
            
            # Remove special characters but keep punctuation
            text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
            
            # Truncate if too long (PEGASUS has 1024 token limit)
            tokens = self.tokenizer.encode(text, truncation=True, max_length=max_length)
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            
            return text
            
        except Exception as e:
            logger.error(f"Error in text preprocessing: {str(e)}")
            return text
    
    def summarize(self, 
                  text: str, 
                  max_length: int = 150, 
                  min_length: int = 50,
                  num_beams: int = 8,
                  length_penalty: float = 2.0,
                  early_stopping: bool = True,
                  temperature: float = 1.0,
                  no_repeat_ngram_size: int = 3) -> Dict[str, Any]:
        """
        Generate factual summary for input text using PEGASUS
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            num_beams: Number of beams for beam search
            length_penalty: Length penalty parameter
            early_stopping: Whether to stop early
            temperature: Sampling temperature (lower = more factual)
            no_repeat_ngram_size: Prevent repetition of n-grams
            
        Returns:
            Dictionary containing summary and metadata
        """
        try:
            # Preprocess the input text
            clean_text = self.preprocess_text(text)
            
            # Calculate input statistics
            input_word_count = len(clean_text.split())
            input_char_count = len(clean_text)
            
            # Adjust max_length if input is too short
            if input_word_count < max_length:
                suggested_max = min(input_word_count, 100)
                st.info(f"ℹ️ Input has {input_word_count} words. Consider reducing max_length to {suggested_max} for optimal results.")
                max_length = min(max_length, suggested_max)
            
            # Generate summary with PEGASUS
            start_time = time.time()
            
            summary_output = self.summarizer(
                clean_text,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                temperature=temperature,
                no_repeat_ngram_size=no_repeat_ngram_size
            )
            
            generation_time = time.time() - start_time
            
            # Get raw summary and clean it
            raw_summary = summary_output[0]['summary_text']
            summary_text = self.clean_summary_text(raw_summary)
            
            # Calculate summary statistics
            summary_word_count = len(summary_text.split())
            summary_char_count = len(summary_text)
            
            # Calculate compression ratio
            compression_ratio = (summary_char_count / input_char_count * 100) if input_char_count > 0 else 0
            
            # Get model info for display
            model_info = self.MODEL_INFO.get(self.model_name, {
                "name": "Unknown",
                "description": "",
                "characteristics": ""
            })
            
            return {
                'summary': summary_text,
                'metadata': {
                    'input_word_count': input_word_count,
                    'input_char_count': input_char_count,
                    'summary_word_count': summary_word_count,
                    'summary_char_count': summary_char_count,
                    'compression_ratio': compression_ratio,
                    'generation_time': generation_time,
                    'model_used': self.model_name,
                    'model_name_display': model_info['name'],
                    'model_description': model_info['description'],
                    'model_characteristics': model_info['characteristics'],
                    'raw_summary': raw_summary  # Keep for debugging
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {
                'summary': f"Error generating summary: {str(e)}",
                'metadata': {'error': str(e)}
            }
    
    def batch_summarize(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Summarize multiple texts in batch
        
        Args:
            texts: List of input texts
            **kwargs: Additional arguments for summarization
            
        Returns:
            List of summary dictionaries with metadata
        """
        summaries = []
        progress_bar = st.progress(0)
        
        for i, text in enumerate(tqdm(texts, desc="Generating summaries")):
            result = self.summarize(text, **kwargs)
            summaries.append(result)
            
            # Update progress
            progress_bar.progress((i + 1) / len(texts))
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.1)
        
        return summaries

class FactualityChecker:
    """
    Simple factuality checker for summaries
    """
    
    @staticmethod
    def check_factual_consistency(original: str, summary: str) -> Dict[str, Any]:
        """
        Check if summary maintains factual consistency with original
        
        Args:
            original: Original text
            summary: Generated summary
            
        Returns:
            Dictionary with factuality metrics
        """
        # Find potential named entities (capitalized words)
        original_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', original))
        summary_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', summary))
        
        # Find numbers and dates
        original_numbers = set(re.findall(r'\b\d+(?:[,.]\d+)?\b', original))
        summary_numbers = set(re.findall(r'\b\d+(?:[,.]\d+)?\b', summary))
        
        # Calculate entity preservation
        preserved_entities = original_entities.intersection(summary_entities)
        entity_preservation = len(preserved_entities) / len(original_entities) if original_entities else 1.0
        
        # Calculate number preservation
        preserved_numbers = original_numbers.intersection(summary_numbers)
        number_preservation = len(preserved_numbers) / len(original_numbers) if original_numbers else 1.0
        
        # Identify potential hallucinations (entities in summary not in original)
        hallucinations = summary_entities - original_entities
        
        # Filter out common abbreviations that might be false positives
        common_abbreviations = {'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Inc', 'Ltd', 'Co', 'US', 'UK', 'EU'}
        filtered_hallucinations = [h for h in hallucinations if h not in common_abbreviations]
        
        return {
            'entity_preservation_ratio': entity_preservation,
            'number_preservation_ratio': number_preservation,
            'preserved_entities': list(preserved_entities),
            'potential_hallucinations': filtered_hallucinations,
            'hallucination_count': len(filtered_hallucinations),
            'factuality_score': (entity_preservation + number_preservation) / 2
        }

class PipelineBuilder:
    """
    Build training and inference pipelines
    """
    
    def __init__(self, summarizer: PegasusSummarizer):
        self.summarizer = summarizer
        
    def build_inference_pipeline(self):
        """
        Build an inference pipeline that abstracts away model complexities
        
        Returns:
            Inference function with simple interface
        """
        def inference_pipeline(text: str, 
                              detail_level: str = "balanced",
                              focus_areas: List[str] = None,
                              **kwargs):
            """
            Simplified inference function with high-level controls
            
            Args:
                text: Input text
                detail_level: "concise", "balanced", or "detailed"
                focus_areas: List of aspects to focus on
                **kwargs: Raw summary parameters
                
            Returns:
                Generated summary with metadata
            """
            
            # Map detail level to parameters
            detail_params = {
                "concise": {"max_length": 80, "min_length": 30, "num_beams": 4},
                "balanced": {"max_length": 150, "min_length": 50, "num_beams": 6},
                "detailed": {"max_length": 250, "min_length": 100, "num_beams": 8}
            }
            
            params = detail_params.get(detail_level, detail_params["balanced"])
            params.update(kwargs)
            
            # Add focus areas to prompt if specified
            if focus_areas:
                focus_text = f" Focus particularly on: {', '.join(focus_areas)}."
                text = text + focus_text
            
            return self.summarizer.summarize(text, **params)
        
        return inference_pipeline
    
    def build_training_pipeline(self):
        """
        Build a training pipeline configuration
        """
        training_config = {
            'model': 'PEGASUS',
            'dataset': 'XSum',
            'training_args': {
                'learning_rate': 5e-5,
                'batch_size': 4,
                'epochs': 3,
                'warmup_steps': 500,
                'weight_decay': 0.01,
                'max_grad_norm': 1.0,
                'gradient_accumulation_steps': 2,
                'evaluation_strategy': 'steps',
                'eval_steps': 500,
                'save_steps': 1000,
                'logging_steps': 100,
                'load_best_model_at_end': True,
                'metric_for_best_model': 'rouge2',
                'greater_is_better': True
            },
            'note': 'Full fine-tuning requires GPU with >16GB memory and several hours of training time'
        }
        
        return training_config

# Initialize session state
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = None
if 'xsum_loader' not in st.session_state:
    st.session_state.xsum_loader = None
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'summary_history' not in st.session_state:
    st.session_state.summary_history = []
if 'fact_checker' not in st.session_state:
    st.session_state.fact_checker = FactualityChecker()
if 'current_sample' not in st.session_state:
    st.session_state.current_sample = None
if 'clear_trigger' not in st.session_state:
    st.session_state.clear_trigger = False

def load_model(model_choice: str):
    """
    Load the PEGASUS summarizer model based on selection
    FIXED: Now properly reloads when selection changes
    """
    model_map = {
        "PEGASUS CNN/DailyMail (Best for News)": "google/pegasus-cnn_dailymail",
        "PEGASUS XSum (Extreme Summarization)": "google/pegasus-xsum",
        "PEGASUS Large (General Purpose)": "google/pegasus-large"
    }
    
    model_name = model_map.get(model_choice, "google/pegasus-cnn_dailymail")
    
    # Check if we need to load a new model
    if st.session_state.current_model != model_name:
        # Clear existing model
        st.session_state.summarizer = None
        st.session_state.current_model = model_name
        
        # Load new model
        with st.spinner(f"Loading {model_choice}... This may take a few minutes..."):
            st.session_state.summarizer = PegasusSummarizer(model_name)
        st.sidebar.success(f"✅ {model_choice} loaded successfully!")
    
    return st.session_state.summarizer

def load_xsum_data(sample_size: int = 100):
    """Load XSum dataset"""
    if st.session_state.xsum_loader is None:
        st.session_state.xsum_loader = XSumDataLoader()
        with st.spinner(f"Loading XSum dataset from Hugging Face ({sample_size} samples)..."):
            st.session_state.xsum_loader.load_dataset(split="test", sample_size=sample_size)
        st.session_state.dataset_loaded = True
    return st.session_state.xsum_loader

def display_model_info(model_choice):
    """Display information about the selected model"""
    model_info = {
        "PEGASUS CNN/DailyMail (Best for News)": {
            "icon": "📰",
            "description": "Trained on CNN/DailyMail news articles",
            "characteristics": "Produces detailed 2-3 sentence summaries",
            "best_for": "Standard news articles, factual retention",
            "example": "Original: 500 words → Summary: 2-3 sentences with key facts"
        },
        "PEGASUS XSum (Extreme Summarization)": {
            "icon": "📋",
            "description": "Trained on BBC XSum dataset",
            "characteristics": "Produces single-sentence extreme summaries",
            "best_for": "Concise headlines, BBC-style summaries",
            "example": "Original: 500 words → Summary: 1 sentence capturing essence"
        },
        "PEGASUS Large (General Purpose)": {
            "icon": "📚",
            "description": "Trained on mixed datasets",
            "characteristics": "Balanced between detail and conciseness",
            "best_for": "General purpose, mixed content types",
            "example": "Original: 500 words → Summary: 2 sentences balanced"
        }
    }
    
    info = model_info.get(model_choice, {})
    
    st.sidebar.markdown(f"""
    <div class="info-box">
        <h4>{info.get('icon', '🤖')} Current Model: {model_choice}</h4>
        <p><strong>Description:</strong> {info.get('description', '')}</p>
        <p><strong>Characteristics:</strong> {info.get('characteristics', '')}</p>
        <p><strong>Best for:</strong> {info.get('best_for', '')}</p>
        <p><strong>Example:</strong> {info.get('example', '')}</p>
    </div>
    """, unsafe_allow_html=True)

def clear_text_input():
    """Function to clear text input by resetting the key"""
    st.session_state.clear_trigger = not st.session_state.clear_trigger
    st.session_state.manual_text = ""
    st.session_state.current_sample = None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📰 PEGASUS News Summarizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Google\'s PEGASUS LLM for Factual, Detailed Summaries</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://huggingface.co/front/assets/huggingface_logo.svg", width=200)
        st.title("⚙️ Configuration")
        st.markdown("---")
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        model_choice = st.selectbox(
            "Choose PEGASUS variant:",
            ["PEGASUS CNN/DailyMail (Best for News)", 
             "PEGASUS XSum (Extreme Summarization)",
             "PEGASUS Large (General Purpose)"],
            index=0,
            key="model_selector",
            help="Select different PEGASUS variants to see how they produce different summary styles"
        )
        
        # Display model information
        display_model_info(model_choice)
        
        # Load model button
        if st.button("🚀 Load/Reload Model", use_container_width=True):
            summarizer = load_model(model_choice)
        
        st.markdown("---")
        
        # Dataset Configuration
        st.subheader("📊 Dataset Settings")
        sample_size = st.slider("XSum Sample Size", 50, 500, 100, 50, key="sample_size")
        
        if st.button("📥 Load XSum Dataset", use_container_width=True):
            xsum_loader = load_xsum_data(sample_size)
        
        st.markdown("---")
        
        # Dataset Statistics (if loaded)
        if st.session_state.dataset_loaded:
            st.subheader("📈 XSum Dataset Stats")
            stats = st.session_state.xsum_loader.get_statistics()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Samples", f"{stats['total_samples']:.0f}")
                st.metric("Avg Doc Length", f"{stats['avg_document_length']:.0f} chars")
                st.metric("Avg Doc Words", f"{stats['doc_word_count_avg']:.0f}")
            with col2:
                st.metric("Avg Summary", f"{stats['avg_summary_length']:.0f} chars")
                st.metric("Avg Summary Words", f"{stats['summary_word_count_avg']:.0f}")
                st.metric("Compression", f"{stats['avg_summary_length']/stats['avg_document_length']*100:.1f}%")
        
        st.markdown("---")
        st.caption("Built with Streamlit & Hugging Face | Google PEGASUS Model")
    
    # Main content area - Show current model status
    if st.session_state.current_model:
        model_display = {
            "google/pegasus-cnn_dailymail": "📰 PEGASUS CNN/DailyMail",
            "google/pegasus-xsum": "📋 PEGASUS XSum",
            "google/pegasus-large": "📚 PEGASUS Large"
        }.get(st.session_state.current_model, "Unknown")
        
        st.markdown(f"""
        <div class="model-indicator">
            ✅ Active Model: {model_display}
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Summarize", 
        "📊 Dataset Analysis", 
        "🔍 Factuality Check",
        "📚 History",
        "ℹ️ About"
    ])
    
    with tab1:
        st.header("Generate Factual Summary")
        
        # Create two columns for input/output
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Article")
            
            # Input method selection
            input_method = st.radio(
                "Choose input method:",
                ["Enter text manually", "Load from XSum dataset", "Upload text file"],
                horizontal=True,
                key="input_method"
            )
            
            text_input = ""
            
            if input_method == "Enter text manually":
                # Use a text area with a unique key that can be updated
                text_input = st.text_area(
                    "Enter your news article:",
                    height=300,
                    placeholder="Paste your news article here...",
                    key=f"manual_text_{st.session_state.clear_trigger}"
                )
                
                # Store in session state for other operations
                st.session_state.manual_text = text_input
                
                # Clear button that doesn't directly modify widget state
                if st.button("📋 Clear", use_container_width=True):
                    st.session_state.clear_trigger = not st.session_state.clear_trigger
                    st.rerun()
                    
            elif input_method == "Load from XSum dataset":
                if st.session_state.dataset_loaded:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("🎲 Random Article", use_container_width=True, key="random_btn"):
                            sample = st.session_state.xsum_loader.get_random_sample()
                            if sample:
                                st.session_state.current_sample = sample['document']
                                st.rerun()
                    
                    with col_b:
                        # Search functionality
                        search_term = st.text_input("Search articles:", placeholder="Enter keyword...", key="search_term")
                        if search_term:
                            results = st.session_state.xsum_loader.search_articles(search_term, 5)
                            if results:
                                selected = st.selectbox("Select article:", 
                                                       [r['document'][:100] + "..." for r in results],
                                                       key="article_selector")
                                if selected:
                                    idx = [r['document'][:100] + "..." for r in results].index(selected)
                                    st.session_state.current_sample = results[idx]['document']
                                    st.rerun()
                    
                    if st.session_state.current_sample:
                        text_input = st.text_area(
                            "XSum Article:",
                            value=st.session_state.current_sample,
                            height=300,
                            key="xsum_display"
                        )
                    else:
                        text_input = st.text_area(
                            "XSum Article:",
                            height=300,
                            placeholder="Click the button to load a random XSum article...",
                            key="xsum_placeholder"
                        )
                else:
                    st.warning("Please load XSum dataset first from the sidebar.")
                    text_input = ""
                    
            else:  # Upload text file
                uploaded_file = st.file_uploader("Choose a text file", type=['txt', 'csv'], key="file_uploader")
                if uploaded_file is not None:
                    text_input = uploaded_file.read().decode()
                    text_input = st.text_area("File content:", value=text_input, height=300, key="file_content")
                else:
                    text_input = ""
            
            # Summary parameters
            with st.expander("⚙️ Advanced Summary Parameters", expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    detail_level = st.select_slider(
                        "Detail Level",
                        options=["concise", "balanced", "detailed"],
                        value="balanced",
                        key="detail_level"
                    )
                    max_length = st.slider("Max Length", 50, 300, 150, 10, key="max_length")
                    num_beams = st.slider("Num Beams", 4, 12, 8, 1, key="num_beams")
                with col_b:
                    min_length = st.slider("Min Length", 20, 150, 50, 10, key="min_length")
                    length_penalty = st.slider("Length Penalty", 0.5, 3.0, 2.0, 0.1, key="length_penalty")
                    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1, 
                                          help="Lower = more factual, Higher = more creative",
                                          key="temperature")
            
            # Focus areas
            focus_areas = st.multiselect(
                "Focus on specific aspects (optional):",
                ["key facts", "numbers/statistics", "names/entities", "timeline", "causes", "consequences"],
                default=[],
                key="focus_areas"
            )
        
        with col2:
            st.subheader("Generated Summary")
            
            if st.button("✨ Generate Factual Summary", type="primary", use_container_width=True, key="generate_btn"):
                # Get the current text input based on method
                if input_method == "Enter text manually":
                    current_text = st.session_state.get('manual_text', '')
                elif input_method == "Load from XSum dataset":
                    current_text = st.session_state.get('current_sample', '')
                else:
                    current_text = text_input
                
                if current_text and len(current_text.strip()) > 0:
                    with st.spinner("PEGASUS is generating a factual summary..."):
                        try:
                            # Ensure model is loaded
                            if st.session_state.summarizer is None:
                                summarizer = load_model(model_choice)
                            else:
                                summarizer = st.session_state.summarizer
                            
                            # Generate summary
                            result = summarizer.summarize(
                                current_text,
                                max_length=max_length,
                                min_length=min_length,
                                num_beams=num_beams,
                                length_penalty=length_penalty,
                                temperature=temperature
                            )
                            
                            summary = result['summary']
                            metadata = result['metadata']
                            
                            # Factuality check
                            fact_check = st.session_state.fact_checker.check_factual_consistency(
                                current_text, summary
                            )
                            
                            # Store in history
                            st.session_state.summary_history.append({
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'input_preview': current_text[:200] + "...",
                                'summary': summary,
                                'metadata': metadata,
                                'fact_check': fact_check,
                                'model_used': metadata.get('model_name_display', 'Unknown')
                            })
                            
                            # Display model info
                            st.info(f"**Model used:** {metadata.get('model_name_display', 'Unknown')} - {metadata.get('model_description', '')}")
                            
                            # Display summary
                            st.success("Summary generated successfully!")
                            st.text_area("Summary:", value=summary, height=200, key="summary_output")
                            
                            # Display metrics in columns
                            col_x, col_y, col_z = st.columns(3)
                            with col_x:
                                st.metric("Original Length", f"{metadata['input_word_count']} words")
                                st.metric("Summary Length", f"{metadata['summary_word_count']} words")
                            with col_y:
                                st.metric("Compression", f"{metadata['compression_ratio']:.1f}%")
                                st.metric("Generation Time", f"{metadata['generation_time']:.2f}s")
                            with col_z:
                                st.metric("Factuality Score", f"{fact_check['factuality_score']:.2f}")
                                st.metric("Keywords Preserved", f"{fact_check['entity_preservation_ratio']:.0%}")
                            
                            # Warning for potential hallucinations
                            if fact_check['hallucination_count'] > 0:
                                st.warning(f"⚠️ Potential hallucinations detected: {fact_check['hallucination_count']} new entities in summary")
                                if fact_check['potential_hallucinations']:
                                    with st.expander("View potential hallucinations"):
                                        st.write(fact_check['potential_hallucinations'])
                            
                            # Download button
                            st.download_button(
                                "📥 Download Summary",
                                summary,
                                file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                key="download_btn"
                            )
                            
                        except Exception as e:
                            st.error(f"Error generating summary: {str(e)}")
                else:
                    st.warning("Please enter some text to summarize.")
    
    with tab2:
        st.header("XSum Dataset Analysis")
        
        if st.session_state.dataset_loaded:
            stats = st.session_state.xsum_loader.get_statistics()
            df = pd.DataFrame({
                'document': st.session_state.xsum_loader.dataset['document'],
                'summary': st.session_state.xsum_loader.dataset['summary']
            })
            df['doc_length'] = df['document'].str.len()
            df['summary_length'] = df['summary'].str.len()
            df['doc_word_count'] = df['document'].str.split().str.len()
            df['summary_word_count'] = df['summary'].str.split().str.len()
            
            # Distribution plots
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Document Length Distribution")
                fig = px.histogram(df, x='doc_word_count', nbins=30, 
                                  title="Document Word Count Distribution",
                                  labels={'doc_word_count': 'Number of Words'})
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Summary Length Distribution")
                fig = px.histogram(df, x='summary_word_count', nbins=30,
                                  title="Summary Word Count Distribution",
                                  labels={'summary_word_count': 'Number of Words'})
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot with manual trend line
            st.subheader("Document vs Summary Length Relationship")
            fig = px.scatter(df, x='doc_word_count', y='summary_word_count',
                           title="Document Length vs Summary Length",
                           labels={'doc_word_count': 'Document Words', 'summary_word_count': 'Summary Words'},
                           opacity=0.6,
                           color_discrete_sequence=['#1E88E5'])
            fig.update_traces(marker=dict(size=6))
            fig.update_layout(
                xaxis_title="Document Length (words)",
                yaxis_title="Summary Length (words)",
                hovermode='closest'
            )
            
            # Add a simple trend line manually using numpy
            if len(df) > 1:
                z = np.polyfit(df['doc_word_count'], df['summary_word_count'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(df['doc_word_count'].min(), df['doc_word_count'].max(), 100)
                y_trend = p(x_trend)
                
                fig.add_trace(go.Scatter(
                    x=x_trend, 
                    y=y_trend,
                    mode='lines',
                    name=f'Trend (slope: {z[0]:.3f})',
                    line=dict(color='red', width=2, dash='dash')
                ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show correlation
            correlation = df['doc_word_count'].corr(df['summary_word_count'])
            st.info(f"📊 **Correlation coefficient:** {correlation:.3f} - This shows the strength of relationship between document and summary lengths.")
            
            # Summary statistics
            st.subheader("Detailed Statistics")
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Avg Document", f"{stats['doc_word_count_avg']:.0f} words")
            with col_b:
                st.metric("Avg Summary", f"{stats['summary_word_count_avg']:.0f} words")
            with col_c:
                st.metric("Compression Ratio", f"{stats['avg_summary_length']/stats['avg_document_length']*100:.1f}%")
            with col_d:
                st.metric("Correlation", f"{correlation:.3f}")
            
            # Sample data table
            st.subheader("Sample Articles")
            sample_df = df[['doc_word_count', 'summary_word_count']].head(10)
            sample_df['document_preview'] = df['document'].str[:100] + "..."
            sample_df['summary_preview'] = df['summary'].str[:100] + "..."
            st.dataframe(
                sample_df[['document_preview', 'summary_preview', 'doc_word_count', 'summary_word_count']],
                use_container_width=True,
                column_config={
                    "document_preview": "Article Preview",
                    "summary_preview": "Summary Preview",
                    "doc_word_count": "Doc Words",
                    "summary_word_count": "Summary Words"
                }
            )
        else:
            st.info("Please load the XSum dataset first from the sidebar.")
    
    with tab3:
        st.header("Factuality Check Analysis")
        
        if st.session_state.summary_history:
            st.subheader("Recent Summaries Factuality")
            
            # Create factuality metrics dataframe
            fact_data = []
            for item in st.session_state.summary_history[-10:]:  # Last 10
                fact_data.append({
                    'Time': item['timestamp'],
                    'Model': item.get('model_used', 'Unknown'),
                    'Factuality Score': item['fact_check']['factuality_score'],
                    'Entity Preservation': item['fact_check']['entity_preservation_ratio'],
                    'Hallucinations': item['fact_check']['hallucination_count']
                })
            
            if fact_data:
                fact_df = pd.DataFrame(fact_data)
                
                # Plot factuality trends
                fig = px.line(fact_df, x='Time', y=['Factuality Score', 'Entity Preservation'],
                            title="Factuality Trends Over Time",
                            labels={'value': 'Score', 'variable': 'Metric'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Hallucination count by model
                fig = px.bar(fact_df, x='Time', y='Hallucinations', color='Model',
                           title="Hallucination Count by Model",
                           labels={'Hallucinations': 'Number of Potential Hallucinations'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Tips for improving factuality
            with st.expander("💡 Tips for Improving Factuality"):
                st.markdown("""
                - **Lower temperature** (0.3-0.5) for more factual outputs
                - **Increase num_beams** (8-12) for better search
                - **Use longer context** - ensure input isn't truncated
                - **Focus on key entities** in your prompt
                - **Verify numbers and dates** in the output
                - **Try different models** - CNN/DailyMail is best for factual news
                """)
        else:
            st.info("Generate some summaries first to see factuality analysis.")
    
    with tab4:
        st.header("Summary History")
        
        if st.session_state.summary_history:
            for i, item in enumerate(reversed(st.session_state.summary_history[-20:])):
                with st.expander(f"Summary {len(st.session_state.summary_history)-i} - {item['timestamp']} ({item.get('model_used', 'Unknown')})"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Input Preview:**")
                        st.text(item['input_preview'])
                    with col_b:
                        st.markdown("**Summary:**")
                        st.text(item['summary'][:200] + "...")
                    
                    # Metrics
                    col_x, col_y, col_z = st.columns(3)
                    with col_x:
                        st.metric("Words", f"{item['metadata']['summary_word_count']}")
                    with col_y:
                        st.metric("Compression", f"{item['metadata']['compression_ratio']:.1f}%")
                    with col_z:
                        st.metric("Factuality", f"{item['fact_check']['factuality_score']:.2f}")
                    
                    # View full buttons
                    if st.button(f"View Full Summary {i}", key=f"view_{i}"):
                        st.info(f"Full summary:\n{item['summary']}")
        else:
            st.info("No summaries generated yet.")
    
    with tab5:
        st.header("About This Application")
        
        st.markdown("""
        ### 📋 Overview
        
        This application fulfills the requirements of the text summarization assignment using **Google's PEGASUS LLM**:
        
        #### 1. Model Selection
        - **Model:** Google PEGASUS with three variants:
          - **CNN/DailyMail**: Best for detailed news summarization
          - **XSum**: Specialized for extreme single-sentence summaries
          - **Large**: General purpose summarization
        - **Why PEGASUS?**
          - Specifically designed for summarization tasks
          - State-of-the-art performance on news datasets
          - Lower hallucination rate than general-purpose LLMs
          - Excellent factual retention
        
        #### 2. Data Preprocessing
        - **Dataset:** XSum from Hugging Face using `load_dataset('xsum')`
        - **Libraries:**
          - `transformers` - For PEGASUS model and tokenizer
          - `datasets` - For XSum dataset loading
          - `streamlit` - For web application
          - `plotly` - For interactive visualizations
        
        #### 3. Pipelines
        - **Training Pipeline:** Configured for fine-tuning
        - **Inference Pipeline:** Abstracted complexity
        - **Factuality Pipeline:** Built-in hallucination detection
        
        #### 4. Web Application
        - **Framework:** Streamlit
        - **Features:**
          - Three PEGASUS variants to compare
          - Multiple input methods
          - Adjustable parameters
          - Real-time factuality checking
          - Summary history tracking
        
        ### 📚 References
        
        - Zhang, J., et al. (2020). PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization
        - Narayan, S., et al. (2018). Don't Give Me the Details, Just the Summary!
        
        """)

if __name__ == "__main__":
    main()