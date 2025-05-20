import logging
import re
import os
import json
import random

logger = logging.getLogger(__name__)

class Agent:
    """Base agent class"""
    def __init__(self, name):
        self.name = name
    
    def process(self, query, *args, **kwargs):
        """Process input data and return output"""
        raise NotImplementedError("Subclasses must implement process method")

class RetrievalAgent(Agent):
    """Agent responsible for retrieving relevant information"""
    def __init__(self, chunks, embedding_processor=None, similarity_threshold=0.4):
        super().__init__("RetrievalAgent")
        self.chunks = chunks
        # We'll use a simple keyword matching approach instead of embeddings
        self.similarity_threshold = similarity_threshold
    
    def process(self, query, *args, **kwargs):
        """Retrieve relevant chunks based on keyword matching"""
        try:
            query_words = self._extract_keywords(query)
            
            # calculate matches with all chunks
            matches = []
            for chunk in self.chunks:
                if chunk.get('text'):
                    # concatenate, None issues 
                    chunk_text = chunk.get('text', '')
                    article_text = chunk.get('article', '')
                    if article_text:
                        chunk_text = chunk_text + ' ' + article_text
                    
                    match_score = self._calculate_keyword_match(query_words, chunk_text)
                    
                    if chunk.get('is_overlap'):
                        match_score *= 1.2 
                    
                    matches.append((chunk, match_score))
            
            # sort scores 
            matches.sort(key=lambda x: x[1], reverse=True)
            
            # get top k  matching 
            relevant_chunks = []
            regular_count = 0
            overlap_count = 0
            
            for chunk, score in matches:
                if score > 0: 
                    if chunk.get('is_overlap'):
                        overlap_count += 1
                    else:
                        regular_count += 1
                    
                    relevant_chunks.append(chunk)
                    
                    # limit to top chunks, but ensure a mix of regular and overlapping
                    # for better diversity in results
                    if (regular_count >= 3 and overlap_count >= 2) or len(relevant_chunks) >= 7:
                        break
            
            if not relevant_chunks:
                logger.warning(f"No relevant chunks found for query: {query}")
                return []
            
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
            return relevant_chunks
        
        except Exception as e:
            logger.exception(f"Error retrieving relevant chunks: {e}")
            return []
    
    def _extract_keywords(self, text):
        """Extract keywords from text"""
        stop_words = ['من', 'في', 'على', 'إلى', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك', 'هو', 'هي', 'أو', 'و']
        words = re.findall(r'\w+', text.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    def _calculate_keyword_match(self, query_words, text):
        """Calculate match score based on keyword presence"""
        text_lower = text.lower()
        matches = sum(1 for word in query_words if word in text_lower)
        return matches / max(1, len(query_words))

class GenerationAgent(Agent):
    """Agent responsible for generating answers"""
    def __init__(self):
        super().__init__("GenerationAgent")
    
    def process(self, query, relevant_chunks, *args, **kwargs):
        """Generate an answer based on query and relevant chunks"""
        try:
            # no relevant chunks found
            if not relevant_chunks:
                return "عذراً، لا يمكنني العثور على معلومات كافية للإجابة على هذا السؤال في قانون العمل الأردني."
            
            # response based on the query and relevant chunks
            answer = self._generate_simple_answer(query, relevant_chunks)
            
            return answer
        
        except Exception as e:
            logger.exception(f"Error generating answer: {e}")
            return "عذراً، حدث خطأ أثناء معالجة السؤال. يرجى المحاولة مرة أخرى."
    
    def _generate_simple_answer(self, query, relevant_chunks):
        """Generate a simple answer for demonstration purposes"""
        # for questions about specific articles
        article_pattern = re.compile(r'ماد[ةه]\s*\(?\s*(\d+)\s*\)?', re.IGNORECASE)
        article_match = article_pattern.search(query)
        
        if article_match:
            article_num = article_match.group(1)
            # find chunks with that article number
            for chunk in relevant_chunks:
                if chunk.get("article") and article_num in chunk.get("article"):
                    if chunk.get("is_overlap"):
                        return f"وفقاً لـ {chunk['article']}: {chunk['text']}"
            
            # Try again
            for chunk in relevant_chunks:
                if chunk.get("article") and article_num in chunk.get("article"):
                    return f"وفقاً لـ {chunk['article']}: {chunk['text']}"
        
        overlapping_chunks = [c for c in relevant_chunks if c.get("is_overlap")]
        
        # Process top chunk
        if overlapping_chunks:
            # Use the most relevant overlapping chunk
            most_relevant = overlapping_chunks[0]
            article = most_relevant.get("article", "")
            
            # Construct answer with reference
            response = f"بناءً على {article} من قانون العمل الأردني: {most_relevant['text']}"
            
            # Clean up the response
            response = response.replace("  ", " ").strip()
            
            return response
        elif relevant_chunks:
            # Fallback to regular chunks if no overlapping 
            most_relevant = relevant_chunks[0]
            article = most_relevant.get("article", "")
            
            # Construct answer + reference
            response = f"بناءً على {article} من قانون العمل الأردني: {most_relevant['text']}"
            
            # clean response
            response = response.replace("  ", " ").strip()
            
            return response
        
        return "لم أتمكن من العثور على إجابة محددة في قانون العمل الأردني."

class AgentChain:
    """Chain of agents to process questions and generate answers"""
    def __init__(self, chunks, embedding_processor=None):
        self.retrieval_agent = RetrievalAgent(chunks)
        self.generation_agent = GenerationAgent()
    
    def process_question(self, question):
        """Process a question through the agent chain"""
        # Step 1: Retrieve relevant chunks
        relevant_chunks = self.retrieval_agent.process(question)
        
        # Step 2: make answer
        answer = self.generation_agent.process(question, relevant_chunks)
        
        return answer, relevant_chunks
