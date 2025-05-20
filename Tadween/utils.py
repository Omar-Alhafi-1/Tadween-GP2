import json
import logging
import os
import re
import traceback
from typing import List, Dict, Any, Tuple, Optional

# Setup logger
logger = logging.getLogger(__name__)

def load_chunks(filename=None):
    """Load labor law chunks from a JSON file"""
    if filename is None:
        filename = "labor_law_chunks.json"
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} labor law chunks from {filename}")
        return chunks
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        
        # Try to find the file in the attached_assets directory
        try:
            # Check for file paths with potential spaces in filename
            # Handle the case with a space in the filename
            if filename == 'labor_law_chunks.json':
                potential_filenames = [
                    os.path.join("attached_assets", filename),
                    os.path.join("attached_assets", "labor_law_chunks .json")  # Note the space before .json
                ]
                for potential_file in potential_filenames:
                    if os.path.exists(potential_file):
                        asset_filename = potential_file
                        logger.info(f"Found file at {asset_filename}")
                        break
                else:
                    raise FileNotFoundError(f"Could not find {filename} in attached_assets")
            else:
                asset_filename = os.path.join("attached_assets", filename)
                
            with open(asset_filename, 'r', encoding='utf-8') as f:
                raw_chunks = json.load(f)
            logger.info(f"Loaded {len(raw_chunks)} labor law chunks from {asset_filename}")
            
            # Transform the chunks to the expected format with metadata
            chunks = []
            for chunk in raw_chunks:
                # Extract article and text directly from the source
                text = chunk.get('text', '')
                article = chunk.get('article')
                chunk_id = chunk.get('chunk_id', 0)
                
                # Set display title based on original article format
                if article:
                    display_title = article
                elif chunk_id == 1 and "اسم القانون" in text:
                    display_title = "اسم القانون"
                else:
                    display_title = f"مصدر {chunk_id}"
                
                logger.info(f"Using original article format: {display_title}")
                
                # Create a new chunk with metadata that preserves the original format
                transformed_chunk = {
                    'text': text,
                    'metadata': {
                        'source': f"chunk_{chunk_id}",
                        'article': article,
                        'display_title': display_title,
                        'chunk_id': chunk_id
                    }
                }
                chunks.append(transformed_chunk)
            
            logger.info(f"Transformed {len(chunks)} chunks to the expected format")
            
            # Create additional overlapping chunks
            additional_chunks = []
            for i in range(len(chunks) - 1):
                combined_text = chunks[i]['text'] + " " + chunks[i+1]['text']
                
                # Get articles directly from both chunks
                article1 = chunks[i]['metadata'].get('display_title', None)
                article2 = chunks[i+1]['metadata'].get('display_title', None)
                
                # Create a combined display title preserving the original format
                if article1 and article2:
                    combined_display = f"{article1} و {article2}"
                elif article1:
                    combined_display = f"{article1} والمصدر التالي"
                elif article2:
                    combined_display = f"المصدر السابق و {article2}"
                else:
                    combined_display = f"مصدر {chunks[i]['metadata'].get('chunk_id', 0)} و {chunks[i+1]['metadata'].get('chunk_id', 0)}"
                
                combined_chunk = {
                    'text': combined_text,
                    'metadata': {
                        'source': f"combined_{i}_{i+1}",
                        'article': combined_display,
                        'display_title': combined_display,
                        'chunk_id': f"combined_{i}_{i+1}"
                    }
                }
                additional_chunks.append(combined_chunk)
            
            logger.info(f"Created {len(additional_chunks)} additional overlapping chunks")
            chunks.extend(additional_chunks)
            
            return chunks
        except Exception as e:
            logger.error(f"Error loading labor law chunks: {str(e)}")
            # Return a minimal set of chunks to allow the system to start
            return [{"text": "لم يتم العثور على نصوص القانون", "metadata": {"source": "error", "article": "unknown"}}]

def process_json_batch(questions: List[str], expected_answers: List[str], 
                       agent_chain, batch_size: int = 10, 
                       is_error_response_fn=None, callback=None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process a batch of questions and expected answers, with memory optimization
    
    Args:
        questions (List[str]): List of questions to process
        expected_answers (List[str]): List of expected answers
        agent_chain: Agent chain to process questions
        batch_size (int): Size of batches to process
        is_error_response_fn (callable): Function to check if a response contains errors
    
    Returns:
        Tuple[List[Dict], Dict]: Results and metrics
    """
    import time
    import random
    import gc
    from evaluation import calculate_bleu_score, calculate_bert_score, calculate_llm_similarity
    
    # Default is_error_response function if none provided
    if is_error_response_fn is None:
        ERROR_INDICATORS = [
            "حدث خطأ أثناء معالجة السؤال",
            "خطأ في معالجة الطلب",
            "حدث خطأ في النظام",
            "You exceeded your current quota", 
            "429",
            "exceeded rate limits"
        ]
        
        def is_error_response(text):
            """Check if the response contains error indicators"""
            return any(indicator in text for indicator in ERROR_INDICATORS)
        
        is_error_response_fn = is_error_response
    
    # Initialize results
    all_results = []
    total_questions = len(questions)
    
    # Error checking
    if total_questions == 0:
        logger.warning("No questions to process")
        return [], {
            'total_questions': 0,
            'average_bleu_score': 0,
            'average_bert_score': 0,
            'average_llm_score': 0,
            'bleu_scores': [],
            'bert_scores': [],
            'llm_scores': []
        }
    
    if len(expected_answers) != total_questions:
        logger.warning(f"Mismatch between questions ({total_questions}) and expected answers ({len(expected_answers)})")
        # Adjust the shorter list to match the longer one (with empty strings)
        if len(questions) > len(expected_answers):
            expected_answers.extend([""] * (len(questions) - len(expected_answers)))
        else:
            questions.extend([""] * (len(expected_answers) - len(questions)))
    
    logger.info(f"Processing {total_questions} questions (batch size: {batch_size})")
    
    # Check if we need batch processing
    if total_questions > batch_size:
        logger.info(f"Using batch processing with {batch_size} questions per batch")
        
        # Initialize lists to collect results across all batches
        all_predictions = []
        all_bleu_scores = []
        all_bert_scores = []
        all_llm_scores = []
        
        # Process in batches
        for batch_start in range(0, total_questions, batch_size):
            batch_end = min(batch_start + batch_size, total_questions)
            batch_questions = questions[batch_start:batch_end]
            batch_expected = expected_answers[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1} ({batch_start+1}-{batch_end} of {total_questions})")
            
            # Process this batch
            batch_predictions = []
            
            for i, question in enumerate(batch_questions):
                absolute_index = batch_start + i
                # Add a significant delay between requests to avoid rate limiting
                if i > 0:
                    # Random delay between 1.0 and 2.0 seconds
                    delay = 1.0 + random.random() * 1.0
                    logger.info(f"Waiting {delay:.2f} seconds before processing next question...")
                    time.sleep(delay)
                
                # Try up to 3 times with increasing delays if we get an error
                max_retries = 3
                retry_count = 0
                prediction = None  # Initialize to avoid reference issues
                
                while retry_count < max_retries:
                    try:
                        logger.debug(f"Processing question {absolute_index+1}/{total_questions}: {question[:50]}...")
                        prediction, _ = agent_chain.process_question(question)
                        
                        # Check if the response contains error indicators
                        if is_error_response_fn(prediction):
                            logger.warning(f"Error response for question {absolute_index+1}, retrying... ({retry_count+1}/{max_retries})")
                            retry_count += 1
                            # Exponential backoff - wait longer each retry
                            retry_delay = 1.0 * (2 ** retry_count)
                            logger.info(f"Retry delay: {retry_delay:.2f} seconds")
                            time.sleep(retry_delay)
                            continue  # Try again
                        else:
                            break  # Success, exit retry loop
                    except Exception as e:
                        logger.error(f"Error processing question {absolute_index+1}: {str(e)}")
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.info(f"Retrying question {absolute_index+1} after error ({retry_count}/{max_retries})")
                            retry_delay = 1.0 * (2 ** retry_count)
                            logger.info(f"Retry delay: {retry_delay:.2f} seconds")
                            time.sleep(retry_delay)
                        else:
                            prediction = "عذراً، حدث خطأ أثناء معالجة هذا السؤال."
                            break
                
                # If we still have no prediction after retries, use a placeholder
                if prediction is None:
                    prediction = "عذراً، تعذر الحصول على إجابة بعد عدة محاولات. يرجى المحاولة مرة أخرى لاحقاً."
                
                batch_predictions.append(prediction)
            
            # Calculate metrics for this batch
            try:
                batch_bleu_scores = []
                batch_bert_scores = []
                batch_llm_scores = []
                
                for j, (pred, exp) in enumerate(zip(batch_predictions, batch_expected)):
                    absolute_j = batch_start + j
                    
                    try:
                        bleu = calculate_bleu_score(pred, exp)
                    except Exception as e:
                        logger.error(f"BLEU score error for item {absolute_j+1}: {str(e)}")
                        bleu = 0
                    
                    try:
                        bert = calculate_bert_score(pred, exp)
                    except Exception as e:
                        logger.error(f"BERT score error for item {absolute_j+1}: {str(e)}")
                        bert = 0
                    
                    try:
                        llm = calculate_llm_similarity(pred, exp)
                    except Exception as e:
                        logger.error(f"LLM similarity error for item {absolute_j+1}: {str(e)}")
                        llm = 0.5  # Default middle score as fallback
                    
                    batch_bleu_scores.append(bleu)
                    batch_bert_scores.append(bert)
                    batch_llm_scores.append(llm)
                    
                    # Create result object for this item
                    result = {
                        'id': absolute_j + 1,
                        'question': questions[absolute_j],
                        'ground_truth': expected_answers[absolute_j],
                        'prediction': pred,
                        'bleu_score': bleu,
                        'bert_score': bert,
                        'llm_score': llm
                    }
                    
                    # Call the callback function if provided to stream progress
                    if callback:
                        # Calculate current progress
                        current_progress = (absolute_j + 1) / total_questions
                        
                        # Calculate current averages for processed items so far
                        current_bleu_scores = all_bleu_scores + batch_bleu_scores[:j+1]
                        current_bert_scores = all_bert_scores + batch_bert_scores[:j+1]  
                        current_llm_scores = all_llm_scores + batch_llm_scores[:j+1]
                        
                        # Create progress metrics
                        current_metrics = {
                            'total_questions': total_questions,
                            'processed_questions': absolute_j + 1,
                            'progress': current_progress,
                            'average_bleu_score': sum(current_bleu_scores) / len(current_bleu_scores) if current_bleu_scores else 0,
                            'average_bert_score': sum(current_bert_scores) / len(current_bert_scores) if current_bert_scores else 0,
                            'average_llm_score': sum(current_llm_scores) / len(current_llm_scores) if current_llm_scores else 0
                        }
                        
                        # Send result and progress metrics through callback
                        callback(result, current_metrics)
                
                # Add this batch's results to the overall results
                all_predictions.extend(batch_predictions)
                all_bleu_scores.extend(batch_bleu_scores)
                all_bert_scores.extend(batch_bert_scores)
                all_llm_scores.extend(batch_llm_scores)
                
                # Log batch metrics
                avg_batch_bleu = sum(batch_bleu_scores) / len(batch_bleu_scores) if batch_bleu_scores else 0
                avg_batch_bert = sum(batch_bert_scores) / len(batch_bert_scores) if batch_bert_scores else 0
                avg_batch_llm = sum(batch_llm_scores) / len(batch_llm_scores) if batch_llm_scores else 0
                
                logger.info(f"Batch {batch_start//batch_size + 1} metrics:" + 
                           f" BLEU={avg_batch_bleu:.3f}," +
                           f" BERT={avg_batch_bert:.3f}," + 
                           f" LLM={avg_batch_llm:.3f}")
                
                # Force garbage collection between batches to free memory
                gc.collect()
                
            except Exception as batch_error:
                logger.error(f"Error in batch {batch_start//batch_size + 1}: {str(batch_error)}")
                traceback.print_exc()
                # Continue processing other batches even if this one fails
                continue
        
        # Create final results
        for i, (q, gt, pred, bleu, bert, llm) in enumerate(zip(
                questions, 
                expected_answers, 
                all_predictions, 
                all_bleu_scores, 
                all_bert_scores, 
                all_llm_scores
            )):
            all_results.append({
                'id': i + 1,
                'question': q,
                'ground_truth': gt,
                'prediction': pred,
                'bleu_score': bleu,
                'bert_score': bert,
                'llm_score': llm
            })
        
        # Calculate overall averages
        if all_bleu_scores and all_bert_scores and all_llm_scores:
            avg_bleu = sum(all_bleu_scores) / len(all_bleu_scores)
            avg_bert = sum(all_bert_scores) / len(all_bert_scores)
            avg_llm = sum(all_llm_scores) / len(all_llm_scores)
            
            # Create combined metrics dictionary
            metrics = {
                'total_questions': len(all_results),
                'average_bleu_score': avg_bleu,
                'average_bert_score': avg_bert,
                'average_llm_score': avg_llm,
                'bleu_scores': all_bleu_scores,
                'bert_scores': all_bert_scores,
                'llm_scores': all_llm_scores
            }
        else:
            # If metrics calculation failed entirely
            logger.error("Failed to calculate valid metrics across all batches")
            metrics = {
                'total_questions': len(all_results),
                'average_bleu_score': 0,
                'average_bert_score': 0,
                'average_llm_score': 0,
                'bleu_scores': [0] * len(all_results),
                'bert_scores': [0] * len(all_results),
                'llm_scores': [0] * len(all_results)
            }
    else:
        # For small datasets, process directly without batching
        logger.info("Processing questions directly (small dataset)")
        
        # Process all questions with the agent chain
        predictions = []
        bleu_scores = []
        bert_scores = []
        llm_scores = []
        
        for i, question in enumerate(questions):
            # Add a significant delay between requests to avoid rate limiting
            if i > 0:
                # Random delay between 1.0 and 2.0 seconds
                delay = 1.0 + random.random() * 1.0
                logger.info(f"Waiting {delay:.2f} seconds before processing next question...")
                time.sleep(delay)
            
            # Try up to 3 times with increasing delays if we get an error
            max_retries = 3
            retry_count = 0
            prediction = None  # Initialize to avoid reference issues
            
            while retry_count < max_retries:
                try:
                    logger.debug(f"Processing question {i+1}/{total_questions}: {question[:50]}...")
                    prediction, _ = agent_chain.process_question(question)
                    
                    # Check if the response contains error indicators
                    if is_error_response_fn(prediction):
                        logger.warning(f"Error response for question {i+1}, retrying... ({retry_count+1}/{max_retries})")
                        retry_count += 1
                        # Exponential backoff - wait longer each retry
                        retry_delay = 1.0 * (2 ** retry_count)
                        logger.info(f"Retry delay: {retry_delay:.2f} seconds")
                        time.sleep(retry_delay)
                        continue  # Try again
                    else:
                        break  # Success, exit retry loop
                except Exception as e:
                    logger.error(f"Error processing question {i+1}: {str(e)}")
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"Retrying question {i+1} after error ({retry_count}/{max_retries})")
                        retry_delay = 1.0 * (2 ** retry_count)
                        logger.info(f"Retry delay: {retry_delay:.2f} seconds")
                        time.sleep(retry_delay)
                    else:
                        prediction = "عذراً، حدث خطأ أثناء معالجة هذا السؤال."
                        break
            
            # If we still have no prediction after retries, use a placeholder
            if prediction is None:
                prediction = "عذراً، تعذر الحصول على إجابة بعد عدة محاولات. يرجى المحاولة مرة أخرى لاحقاً."
            
            predictions.append(prediction)
            
            # Calculate metrics for this prediction
            try:
                bleu = calculate_bleu_score(prediction, expected_answers[i])
            except Exception as e:
                logger.error(f"BLEU score error for item {i+1}: {str(e)}")
                bleu = 0
            
            try:
                bert = calculate_bert_score(prediction, expected_answers[i])
            except Exception as e:
                logger.error(f"BERT score error for item {i+1}: {str(e)}")
                bert = 0
            
            try:
                llm = calculate_llm_similarity(prediction, expected_answers[i])
            except Exception as e:
                logger.error(f"LLM similarity error for item {i+1}: {str(e)}")
                llm = 0.5  # Default middle score as fallback
            
            bleu_scores.append(bleu)
            bert_scores.append(bert)
            llm_scores.append(llm)
            
            # Create result object
            result = {
                'id': i + 1,
                'question': question,
                'ground_truth': expected_answers[i],
                'prediction': prediction,
                'bleu_score': bleu,
                'bert_score': bert,
                'llm_score': llm
            }
            
            # Call the callback function if provided to stream progress
            if callback:
                # Calculate current progress
                current_progress = (i + 1) / total_questions
                
                # Calculate current averages
                current_metrics = {
                    'total_questions': total_questions,
                    'processed_questions': i + 1,
                    'progress': current_progress,
                    'average_bleu_score': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0,
                    'average_bert_score': sum(bert_scores) / len(bert_scores) if bert_scores else 0,
                    'average_llm_score': sum(llm_scores) / len(llm_scores) if llm_scores else 0
                }
                
                # Send result and progress metrics through callback
                callback(result, current_metrics)
        
        # Create results list
        for i, (q, gt, pred, bleu, bert, llm) in enumerate(zip(
                questions, 
                expected_answers, 
                predictions, 
                bleu_scores, 
                bert_scores, 
                llm_scores
            )):
            all_results.append({
                'id': i + 1,
                'question': q,
                'ground_truth': gt,
                'prediction': pred,
                'bleu_score': bleu,
                'bert_score': bert,
                'llm_score': llm
            })
        
        # Calculate metrics
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        avg_bert = sum(bert_scores) / len(bert_scores) if bert_scores else 0
        avg_llm = sum(llm_scores) / len(llm_scores) if llm_scores else 0
        
        metrics = {
            'total_questions': len(all_results),
            'average_bleu_score': avg_bleu,
            'average_bert_score': avg_bert,
            'average_llm_score': avg_llm,
            'bleu_scores': bleu_scores,
            'bert_scores': bert_scores,
            'llm_scores': llm_scores
        }
    
    return all_results, metrics