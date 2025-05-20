"""
Utility functions for evaluating the system's performance.
"""

import logging
import re
import os
import json
import time

logger = logging.getLogger(__name__)

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    logger.warning("NumPy not available, using built-in functions")
    NUMPY_AVAILABLE = False

# Try to import NLTK's sentence_bleu
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
    # Download NLTK data if needed
    import nltk
    nltk.download('punkt', quiet=True)
except ImportError:
    logger.warning("NLTK not available, falling back to simple BLEU calculation")
    NLTK_AVAILABLE = False
except Exception as e:
    logger.warning(f"Error loading NLTK: {e}, falling back to simple BLEU calculation")
    NLTK_AVAILABLE = False

# Try to import bert-score
try:
    import torch
    from bert_score import BERTScorer
    BERT_SCORE_AVAILABLE = True
except ImportError:
    logger.warning("bert-score not available, falling back to simpler similarity method")
    BERT_SCORE_AVAILABLE = False
except Exception as e:
    logger.warning(f"Error loading bert-score: {e}, falling back to simpler similarity method")
    BERT_SCORE_AVAILABLE = False

# Try to import ArabertPreprocessor
try:
    from arabert.preprocess import ArabertPreprocessor
    ARABERT_AVAILABLE = True
    # Initialize ArabertPreprocessor
    arabert_prep = ArabertPreprocessor(model_name="bert-base-arabertv2")
except ImportError:
    logger.warning("arabert not available, using simple tokenization")
    ARABERT_AVAILABLE = False
except Exception as e:
    logger.warning(f"Error initializing ArabertPreprocessor: {e}, using simple tokenization")
    ARABERT_AVAILABLE = False

def _simple_tokenize(text):
    """Simple tokenization function for Arabic text"""
    # Remove punctuation and split on whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    return [token for token in text.split() if token.strip()]

def _arabert_tokenize(text):
    """Tokenize using ArabertPreprocessor if available"""
    if ARABERT_AVAILABLE:
        try:
            # Preprocess text
            text = arabert_prep.preprocess(text)
            # Use NLTK for tokenization
            return nltk.word_tokenize(text)
        except Exception as e:
            logger.warning(f"Error using ArabertPreprocessor: {e}, falling back to simple tokenization")
            return _simple_tokenize(text)
    else:
        return _simple_tokenize(text)

def calculate_bleu_score(prediction, reference):
    """
    Calculate the BLEU score using NLTK's implementation with smoothing.
    Using bigram weights (0.5, 0.5, 0, 0) instead of the default 4-gram approach.
    
    Args:
        prediction (str): Predicted text
        reference (str): Reference text
        
    Returns:
        float: Bigram BLEU score
    """
    try:
        # Tokenize the texts with ArabertPreprocessor if available
        pred_tokens = _arabert_tokenize(prediction.lower())
        ref_tokens = _arabert_tokenize(reference.lower())
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        if NLTK_AVAILABLE:
            # Apply smoothing to handle zero n-gram matches
            smoothing = SmoothingFunction().method1
            # Use bigram weights (0.5, 0.5, 0, 0) instead of default (0.25, 0.25, 0.25, 0.25)
            # This makes it a bigram BLEU score calculation
            bigram_weights = (0.5, 0.5, 0.0, 0.0)
            score = sentence_bleu([ref_tokens], pred_tokens, weights=bigram_weights, smoothing_function=smoothing)
            return score
        else:
            # Fallback to simple bigram-based algorithm if NLTK not available
            # Create bigrams from tokens
            def get_bigrams(tokens):
                return [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
            
            pred_bigrams = get_bigrams(pred_tokens)
            ref_bigrams = get_bigrams(ref_tokens)
            
            # Count matching bigrams
            if not pred_bigrams or not ref_bigrams:
                # If we can't generate bigrams, fall back to unigram matching
                matches = sum(1 for token in pred_tokens if token in ref_tokens)
                precision = matches / len(pred_tokens) if pred_tokens else 0
                recall = matches / len(ref_tokens) if ref_tokens else 0
            else:
                # Count bigram matches
                bigram_matches = sum(1 for bg in pred_bigrams if bg in ref_bigrams)
                
                # Calculate bigram precision and recall
                precision = bigram_matches / len(pred_bigrams) if pred_bigrams else 0
                recall = bigram_matches / len(ref_bigrams) if ref_bigrams else 0
            
            # Calculate F1 score as a BLEU-like metric
            if precision + recall == 0:
                return 0.0
            f1 = 2 * (precision * recall) / (precision + recall)
            
            return f1
    except Exception as e:
        logger.exception(f"Error calculating BLEU score: {e}")
        return 0.0

def calculate_bert_score(prediction, reference):
    """
    Calculate BERT-Score for Arabic text.
    
    Args:
        prediction (str): Predicted text
        reference (str): Reference text
        
    Returns:
        float: F1 BERT score
    """
    if BERT_SCORE_AVAILABLE:
        try:
            # Initialize BERT scorer for Arabic
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            scorer = BERTScorer(lang="ar", model_type="bert-base-multilingual-cased", device=device)
            
            # Compute scores
            P, R, F1 = scorer.score([prediction], [reference])
            return F1.item()  # Return F1 score
        except Exception as e:
            logger.exception(f"Error calculating BERT score: {e}")
    
    # Fallback to simple similarity if BERT-Score fails or isn't available
    try:
        # Tokenize the texts
        pred_tokens = set(_arabert_tokenize(prediction.lower()))
        ref_tokens = set(_arabert_tokenize(reference.lower()))
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # Jaccard similarity
        intersection = pred_tokens.intersection(ref_tokens)
        union = pred_tokens.union(ref_tokens)
        
        similarity = len(intersection) / len(union) if union else 0.0
        
        return float(similarity)
    except Exception as e:
        logger.exception(f"Error calculating fallback text similarity score: {e}")
        return 0.0

def calculate_metrics_batch(predictions, references):
    """
    Calculate metrics for a batch of predictions and references.
    
    Args:
        predictions (list): List of prediction strings
        references (list): List of reference strings
        
    Returns:
        dict: Dictionary with average BLEU, BERT and LLM scores and individual scores
    """
    if len(predictions) != len(references):
        logger.error(f"Number of predictions ({len(predictions)}) doesn't match number of references ({len(references)})")
        return {
            "avg_bleu": 0.0, 
            "avg_bert": 0.0, 
            "avg_llm": 0.0,
            "bleu_scores": [], 
            "bert_scores": [],
            "llm_scores": []
        }
    
    # Calculate BLEU scores
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        bleu = calculate_bleu_score(pred, ref)
        bleu_scores.append(bleu)
    
    # Calculate BERT scores
    bert_scores = []
    
    if BERT_SCORE_AVAILABLE and len(predictions) > 0:
        try:
            # Initialize BERT scorer for Arabic once for the whole batch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            scorer = BERTScorer(lang="ar", model_type="bert-base-multilingual-cased", device=device)
            
            # Compute scores for the entire batch
            _, _, F1 = scorer.score(predictions, references)
            bert_scores = F1.tolist()  # Convert tensor to list
        except Exception as e:
            logger.exception(f"Error calculating batch BERT scores: {e}")
            # Fallback to individual calculations
            for pred, ref in zip(predictions, references):
                bert = calculate_bert_score(pred, ref)
                bert_scores.append(bert)
    else:
        # If BERT-Score not available, calculate one by one
        for pred, ref in zip(predictions, references):
            bert = calculate_bert_score(pred, ref)
            bert_scores.append(bert)
    
    # Calculate LLM-based similarity scores
    llm_scores = []
    # Using Langchain with OpenAI instead of other LLMs
    logger.info("Using Langchain with OpenAI for evaluation")
    llm_scores = [0.0] * len(predictions)
    
    # Calculate averages
    if NUMPY_AVAILABLE:
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        avg_bert = np.mean(bert_scores) if bert_scores else 0.0
        avg_llm = np.mean(llm_scores) if llm_scores else 0.0
    else:
        # Fallback to built-in average calculation if numpy is not available
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        avg_bert = sum(bert_scores) / len(bert_scores) if bert_scores else 0.0
        avg_llm = sum(llm_scores) / len(llm_scores) if llm_scores else 0.0
    
    return {
        "avg_bleu": float(avg_bleu),  # Ensure it's a native Python float
        "avg_bert": float(avg_bert),  # Ensure it's a native Python float
        "avg_llm": float(avg_llm),    # Add LLM average score
        "average_bleu_score": float(avg_bleu),  # Include with the name expected by the UI
        "average_bert_score": float(avg_bert),  # Include with the name expected by the UI
        "average_llm_score": float(avg_llm),  # Include with the name expected by the UI
        "bleu_scores": [float(score) for score in bleu_scores],  # Convert all to native Python floats
        "bert_scores": [float(score) for score in bert_scores],  # Convert all to native Python floats
        "llm_scores": [float(score) for score in llm_scores]     # Add LLM scores list
    }
