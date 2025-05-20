from datetime import datetime
import logging
import json
import os
import uuid
from flask import jsonify, request, redirect, url_for, render_template, flash, current_app
from flask_login import login_user, logout_user, login_required, current_user

from app import db
from models import Conversation, Message, User, ChatHistory, History
from agent_chain import EnhancedAgentChain, get_chunks

# Set up logging
logger = logging.getLogger(__name__)

# Constants
LABOR_LAW_JSON = os.path.join('static', 'docs', 'labor_law.json')

@current_app.route('/api/status')
def api_status():
    """API status check endpoint to verify connectivity"""
    logger.info(f"[API Status] Request from {request.remote_addr}")
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.utcnow().isoformat(),
        'app_version': '1.0.1',
        'message': 'API is working properly'
    })

@current_app.route('/api/ajax-test', methods=['GET', 'POST'])
def api_ajax_test():
    """Debug endpoint for AJAX testing"""
    logger.info(f"[AJAX Test] Request from {request.remote_addr} with method {request.method}")
    
    try:
        # Log request headers
        logger.info(f"[AJAX Test] Headers: {dict(request.headers)}")
        
        if request.method == 'POST':
            # Try to get JSON data
            data = request.get_json(silent=True)
            if data:
                logger.info(f"[AJAX Test] Received JSON data: {data}")
                return jsonify({
                    'status': 'success',
                    'message': 'AJAX POST test successful with JSON data',
                    'received_data': data,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            # Try to get form data
            form_data = request.form.to_dict()
            if form_data:
                logger.info(f"[AJAX Test] Received form data: {form_data}")
                return jsonify({
                    'status': 'success',
                    'message': 'AJAX POST test successful with form data',
                    'received_data': form_data,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            # No data
            logger.warning("[AJAX Test] No data found in POST request")
            return jsonify({
                'status': 'warning',
                'message': 'AJAX POST test received, but no data found',
                'timestamp': datetime.utcnow().isoformat()
            })
        
        # GET request
        return jsonify({
            'status': 'success',
            'message': 'AJAX GET test successful',
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logger.error(f"[AJAX Test] Error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Error during AJAX test: {str(e)}',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@current_app.route('/debug')
def debug_page():
    """Debug page for troubleshooting API issues"""
    logger.info(f"[Debug] Loading debug page")
    return current_app.send_static_file('debug.html')
    
@current_app.route('/debug/simple')
def debug_simple():
    """Ultra simplified debugging page for the chat interface"""
    logger.info(f"[Debug] Loading simplified debug interface")
    return current_app.send_static_file('debug_simple.html')

@current_app.route('/debug/chat')
def debug_chat():
    """Debug page for chat interface without authentication requirement"""
    logger.info(f"[Debug] Loading chat interface for debugging without auth")
    try:
        return current_app.send_static_file('debug_chat.html')
    except Exception as e:
        try:
            # Fallback to the regular template
            from flask import render_template
            return render_template('chat_simplified.html')
        except Exception as inner_e:
            logger.error(f"[Debug] Error loading chat template: {str(inner_e)}")
            return f"Error loading chat template: {str(e)} / {str(inner_e)}", 500

@current_app.route('/api/debug/fix-chat', methods=['GET'])
def debug_fix_chat():
    """Create a fresh conversation for debugging purposes without auth"""
    logger.info(f"[Debug] Creating a fresh conversation for debugging")
    
    try:
        # Get or create a test user
        from models import User, Conversation
        from werkzeug.security import generate_password_hash
        
        test_user = User.query.filter_by(username='test_user').first()
        if not test_user:
            logger.info("[Debug] Creating test user")
            test_user = User(username='test_user')
            test_user.password_hash = generate_password_hash('password123')
            db.session.add(test_user)
            db.session.commit()
            logger.info(f"[Debug] Created test user with ID: {test_user.id}")
        
        # Create a new conversation
        conversation = Conversation(user_id=test_user.id)
        db.session.add(conversation)
        db.session.commit()
        logger.info(f"[Debug] Created new conversation with ID: {conversation.id}")
        
        return jsonify({
            'status': 'ok',
            'message': 'Fresh conversation created for debugging',
            'conversation': {
                'id': conversation.id,
                'user_id': conversation.user_id,
                'title': 'محادثة تجريبية',
                'created_at': conversation.created_at.isoformat()
            },
            'instructions': 'Go to /chat after logging in to see the conversation'
        })
    
    except Exception as e:
        logger.error(f"[Debug] Error fixing chat: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Failed to fix chat: {str(e)}'
        }), 500

@current_app.route('/api/debug/conversations', methods=['GET'])
def debug_conversations():
    """Debug endpoint to check conversation data without auth"""
    logger.info(f"[Debug] Checking conversation data structure")
    
    try:
        # Get total counts
        total_users = User.query.count()
        total_conversations = Conversation.query.count()
        total_messages = Message.query.count()
        
        # Get a sample conversation if available (latest one)
        latest_conversation = Conversation.query.order_by(Conversation.updated_at.desc()).first()
        sample_data = None
        
        if latest_conversation:
            message_count = Message.query.filter_by(conversation_id=latest_conversation.id).count()
            sample_data = {
                'id': latest_conversation.id,
                'user_id': latest_conversation.user_id,
                'title': latest_conversation.title or 'محادثة جديدة',
                'created_at': latest_conversation.created_at.isoformat(),
                'updated_at': latest_conversation.updated_at.isoformat(),
                'message_count': message_count
            }
        
        return jsonify({
            'status': 'ok',
            'counts': {
                'users': total_users,
                'conversations': total_conversations,
                'messages': total_messages
            },
            'sample_conversation': sample_data
        })
    
    except Exception as e:
        logger.error(f"[Debug] Error checking database: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Database error: {str(e)}'
        }), 500
        
@current_app.route('/api/debug/llm-eval', methods=['GET'])
def debug_llm_evaluation():
    """Debug endpoint to test LLM evaluation functionality"""
    logger.info(f"[Debug] Testing LLM evaluation functionality")
    
    try:
        # Import directly here to avoid circular imports
        from evaluation import calculate_bert_score
        
        text1 = "هذا نص تجريبي للاختبار"
        text2 = "هذا نص اختباري للتجربة"
        
        result = {
            'langchain_available': True,
            'text1': text1,
            'text2': text2
        }
        
        try:
            # Use BERT score as a replacement for LLM similarity
            similarity = calculate_bert_score(text1, text2)
            result['similarity'] = similarity
            result['success'] = True
        except Exception as e:
            logger.error(f"[Debug] Error running BERT similarity: {str(e)}", exc_info=True)
            result['error'] = str(e)
            result['success'] = False
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"[Debug] Error in LLM evaluation debug endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@current_app.route('/admin', methods=['GET'])
def admin_login_page():
    """Admin login page for accessing debug interface"""
    # Check if user is already logged in
    from flask_login import current_user
    
    if current_user.is_authenticated:
        # If already logged in as admin, redirect to debug page
        if current_user.username == 'test' or current_user.id == 1:
            return redirect(url_for('admin_debug_page'))
    
    logger.info(f"[Admin] Login page accessed from {request.remote_addr}")
    return render_template('admin_login.html')

@current_app.route('/admin_debug', methods=['GET'])
def admin_debug_page():
    """Admin debug page for accessing debug interface"""
    # Check if user is already logged in
    from flask_login import current_user
    
    if not current_user.is_authenticated:
        return redirect(url_for('admin_login_page'))
    
    if current_user.username != 'test' and current_user.id != 1:
        return redirect(url_for('index'))
    
    logger.info(f"[Admin] Debug page accessed from {request.remote_addr}")
    return render_template('admin_debug.html')

@current_app.route('/admin_debug_simple', methods=['GET'])
def admin_debug_simple_page():
    """Admin debug page for accessing debug interface"""
    # Check if user is already logged in
    from flask_login import current_user
    
    if not current_user.is_authenticated:
        return redirect(url_for('admin_login_page'))
    
    if current_user.username != 'test' and current_user.id != 1:
        return redirect(url_for('index'))
    
    logger.info(f"[Admin] Debug simple page accessed from {request.remote_addr}")
    return render_template('admin_debug_simple.html')

@current_app.route('/admin_debug_direct', methods=['GET', 'POST'])
def admin_debug_direct():
    """Admin debug page for direct evaluation"""
    # Check if user is already logged in
    from flask_login import current_user
    
    if not current_user.is_authenticated:
        return redirect(url_for('admin_login_page'))
    
    if current_user.username != 'test' and current_user.id != 1:
        return redirect(url_for('index'))
    
    logger.info(f"[Admin] Debug direct page accessed from {request.remote_addr}")
    return render_template('admin_debug_direct.html')

@current_app.route('/admin_debug_api', methods=['POST'])
def admin_debug_api():
    """Admin debug API for direct evaluation"""
    # Check if user is already logged in
    from flask_login import current_user
    
    if not current_user.is_authenticated:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if current_user.username != 'test' and current_user.id != 1:
        return jsonify({'error': 'Unauthorized'}), 401
    
    logger.info(f"[Admin] Debug API accessed from {request.remote_addr}")
    
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get the question
        question = data.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Initialize agent chain
        agent_chain = EnhancedAgentChain(get_chunks())
        
        # Process the question
        try:
            answer, used_chunks = agent_chain.process_question(question)
            
            # Format sources from chunks for the frontend
            sources = []
            if used_chunks:
                for chunk in used_chunks:
                    if isinstance(chunk, dict) and 'text' in chunk:
                        metadata = chunk.get('metadata', {})
                        article = metadata.get('article') if isinstance(metadata, dict) else None
                        source = metadata.get('source', '') if isinstance(metadata, dict) else ''
                        
                        # Use the display_title from metadata directly
                        if isinstance(metadata, dict) and 'display_title' in metadata and metadata['display_title']:
                            display_title = metadata['display_title']
                        # If no display_title, use the article field directly
                        elif article:
                            display_title = article
                        # For null article values, fallback to a source identifier
                        else:
                            display_title = 'مصدر قانوني'
                        
                        sources.append({
                            'text': chunk['text'].strip(),
                            'article': display_title,
                            'source': source,
                            'metadata': metadata
                        })
            
            return jsonify({
                'answer': answer,
                'sources': sources,
                'debug': {
                    'has_sources': len(sources) > 0,
                    'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
                }
            })
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return jsonify({
                'error': f'Error processing question: {str(e)}'
            }), 500
    except Exception as e:
        logger.error(f"Error in admin debug API: {str(e)}")
        return jsonify({
            'error': f'Error in admin debug API: {str(e)}'
        }), 500

@current_app.route('/api/chat/history', methods=['GET'])
@login_required
def get_chat_history():
    """Get chat history for the current user"""
    try:
        history = ChatHistory.query.filter_by(user_id=current_user.id).order_by(ChatHistory.created_at.desc()).all()
        return jsonify({
            'status': 'success',
            'history': [
                {
                    'id': item.id,
                    'query': item.query,
                    'answer': item.response,
                    'timestamp': item.created_at.isoformat(),
                    'sources': json.loads(item.sources) if item.sources else [],
                    'metrics': json.loads(item.metrics) if item.metrics else {}
                }
                for item in history
            ]
        })
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@current_app.route('/api/chat/clear', methods=['POST'])
@login_required
def clear_chat_history():
    """Clear chat history for the current user"""
    try:
        ChatHistory.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        return jsonify({
            'status': 'success',
            'message': 'Chat history cleared successfully'
        })
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@current_app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.utcnow().isoformat()
    })

@current_app.route('/ask', methods=['POST'])
def ask():
    """Endpoint for handling questions with conversation context support"""
    data = request.json
    question = data.get('question', '')
    conversation_id = data.get('conversation_id')
    
    if not question:
        return jsonify({'error': 'الرجاء إدخال سؤال'}), 400
    
    # Save question to history if user is authenticated
    if current_user.is_authenticated:
        history_entry = History()
        history_entry.user_id = current_user.id
        history_entry.user_query = question  # Use user_query instead of question
        history_entry.session_id = str(uuid.uuid4())  # Generate a random session ID
        db.session.add(history_entry)
        db.session.commit()
    
    # Initialize agent chain
    agent_chain = EnhancedAgentChain(get_chunks())
    
    # Process the question based on conversation context
    try:
        # Get conversation history if conversation_id is provided
        conversation_history = []
        
        if conversation_id and current_user.is_authenticated:
            conversation = Conversation.query.filter_by(id=conversation_id, user_id=current_user.id).first()
            
            if conversation:
                messages = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.timestamp).all()
                
                for msg in messages:
                    conversation_history.append({
                        'content': msg.content,
                        'is_user': msg.is_user
                    })
        
        logger.info(f"Processing question: '{question}'")
        
        # Error message indicating API overload
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
        
        # Process the question with the agent chain
        answer, used_chunks = agent_chain.process_question(question)
        
        # Check if the response contains error indicators
        if is_error_response(answer):
            logger.warning(f"Error in AI response: {answer}")
            fallback_chunk = {
                'text': 'قانون العمل الأردني رقم 8 لسنة 1996 وتعديلاته هو القانون الرئيسي المنظم لعلاقات العمل في الأردن.',
                'article': 'قانون العمل الأردني',
                'source': 'قانون العمل الأردني',
                'metadata': {
                    'article': 'قانون العمل', 
                    'source': 'قانون العمل الأردني',
                    'display_title': 'قانون العمل الأردني رقم 8 لسنة 1996'
                }
            }
            return jsonify({
                'answer': "عذراً، حدث خطأ أثناء معالجة السؤال. يرجى المحاولة مرة أخرى لاحقاً.",
                'chunks': [fallback_chunk],
                'chunk_count': 1,
                'debug': {'error': True, 'reason': 'AI error indicators detected', 'has_chunks': True}
            })
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        fallback_chunk = {
            'text': 'قانون العمل الأردني رقم 8 لسنة 1996 وتعديلاته هو القانون الرئيسي المنظم لعلاقات العمل في الأردن.',
            'article': 'قانون العمل الأردني',
            'source': 'قانون العمل الأردني',
            'metadata': {
                'article': 'قانون العمل', 
                'source': 'قانون العمل الأردني',
                'display_title': 'قانون العمل الأردني رقم 8 لسنة 1996'
            }
        }
        return jsonify({
            'answer': "عذراً، حدث خطأ أثناء معالجة السؤال. يرجى المحاولة مرة أخرى لاحقاً.",
            'chunks': [fallback_chunk],
            'chunk_count': 1,
            'debug': {'error': True, 'reason': f'Exception: {str(e)}', 'has_chunks': True}
        })
    
    # Update history with answer if user is authenticated
    if current_user.is_authenticated:
        history_entry = History.query.filter_by(user_id=current_user.id).order_by(History.timestamp.desc()).first()
        
        if history_entry and history_entry.user_query == question:
            history_entry.answer = answer
            history_entry.referenced_chunks = json.dumps([chunk['text'] for chunk in used_chunks])
            db.session.commit()
    
    # If conversation_id is provided and user is authenticated, save the interaction
    if conversation_id and current_user.is_authenticated:
        conversation = Conversation.query.filter_by(id=conversation_id, user_id=current_user.id).first()
        
        if conversation:
            # Create user message
            user_message = Message()
            user_message.content = question
            user_message.is_user = True
            user_message.conversation_id = conversation_id
            db.session.add(user_message)
            
            # Create system response
            system_message = Message()
            system_message.content = answer
            system_message.is_user = False
            system_message.conversation_id = conversation_id
            db.session.add(system_message)
            
            # Update conversation timestamp
            conversation.updated_at = datetime.utcnow()
            
            db.session.commit()
    
    # Return the answer and used chunks with all metadata
    chunks_for_response = []
    for chunk in used_chunks:
        # Ensure metadata is always populated
        if not chunk.get('metadata'):
            chunk['metadata'] = {}
            
        # If metadata doesn't have article but article field exists, copy it to metadata
        if not chunk['metadata'].get('article') and chunk.get('article'):
            chunk['metadata']['article'] = chunk['article']
            
        # Create a full chunk representation with all metadata
        chunk_obj = {
            'text': chunk['text'],
            'article': chunk.get('article', ''),  # Include article field for backward compatibility
            'source': chunk['metadata'].get('source', '') if isinstance(chunk.get('metadata'), dict) else '',
            'metadata': chunk.get('metadata', {})  # Include all metadata
        }
        
        # Make sure display_title exists in metadata
        if not chunk_obj['metadata'].get('display_title'):
            if chunk_obj['metadata'].get('article'):
                chunk_obj['metadata']['display_title'] = chunk_obj['metadata']['article']
            elif chunk_obj['article']:
                chunk_obj['metadata']['display_title'] = chunk_obj['article']
            else:
                chunk_obj['metadata']['display_title'] = 'مصدر قانوني'
                
        chunks_for_response.append(chunk_obj)
        
    # Add debug info to help troubleshoot
    return jsonify({
        'answer': answer,
        'chunks': chunks_for_response,
        'chunk_count': len(chunks_for_response),
        'debug': {
            'has_chunks': len(chunks_for_response) > 0,
            'sample_chunk': chunks_for_response[0] if chunks_for_response else None
        }
    })