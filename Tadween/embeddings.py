import logging
import os
import re

logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    """
    Simplified class that handles text processing for labor law chunks
    This version doesn't use actual embeddings but provides a compatible interface
    for the rest of the system
    """
    
    def __init__(self, model_name=None):
        """
        Initialize the text processor
        
        Args:
            model_name (str): Not used in this simplified version
        """
        logger.info("Initialized simplified text processor (no embeddings)")
    
    def get_embedding(self, text):
        """
        Get a simplified representation of the text
        
        Args:
            text (str): Text to process
            
        Returns:
            list: List of keywords from the text
        """
        try:
            # Remove null characters and normalize whitespace
            text = text.replace('\x00', '').strip()
            # Extract keywords (simplified)
            return self._extract_keywords(text)
        except Exception as e:
            logger.exception(f"Error processing text: {e}")
            return []
    
    def _extract_keywords(self, text):
        """Extract keywords from text"""
        # Remove stop words
        stop_words = ['من', 'في', 'على', 'إلى', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك', 'هو', 'هي', 'أو', 'و']
        words = re.findall(r'\w+', text.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    def calculate_similarity(self, keywords1, keywords2):
        """
        Calculate a similarity score between two sets of keywords
        
        Args:
            keywords1 (list): First list of keywords
            keywords2 (list): Second list of keywords
            
        Returns:
            float: Similarity score
        """
        try:
            if not keywords1 or not keywords2:
                return 0.0
            
            # Count overlapping keywords
            intersection = set(keywords1) & set(keywords2)
            # Jaccard similarity
            similarity = len(intersection) / (len(set(keywords1) | set(keywords2)))
            
            return float(similarity)
        except Exception as e:
            logger.exception(f"Error calculating similarity: {e}")
            return 0.0
    
    def process_chunks(self, chunks):
        """
        Process labor law chunks and add keyword representations
        
        Args:
            chunks (list): List of labor law chunks
            
        Returns:
            list: List of processed chunks with keyword extraction
        """
        processed_chunks = []
        
        for chunk in chunks:
            try:
                # Clone the chunks 
                processed_chunk = dict(chunk)
                
                # Combine article, text (better context)
                text_to_process = f"{chunk.get('article', '')} {chunk.get('text', '')}"
                
                # No actual embeddings,simply stores the original chunk
                processed_chunks.append(processed_chunk)
            except Exception as e:
                logger.exception(f"Error processing chunk: {e}")
        
        logger.info(f"Processed {len(processed_chunks)} chunks")
        return processed_chunks
