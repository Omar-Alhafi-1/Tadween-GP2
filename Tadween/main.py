import json
import logging
import os
import sys
import time
import random
import gc
import re
import csv
import io
import codecs
import traceback
import uuid
from datetime import datetime
from functools import wraps

# Configure logger
logger = logging.getLogger(__name__)

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, Response, stream_with_context
from flask_login import LoginManager, current_user, login_user, logout_user, login_required, UserMixin
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from agents import AgentChain
from enhanced_agents import EnhancedAgentChain, SpellCorrectionAgent, QueryAnalyzerAgent, RetrievalAgent
from enhanced_agents import RerankerAgent, AnswerGenerationAgent, ConflictDetectionAgent, ReferenceProcessor
from enhanced_agents import RelevanceDetectionAgent, CreativeInterpretationAgent, DisclaimerAgent, StructuringAgent
from embeddings import EmbeddingProcessor
from evaluation import calculate_bleu_score, calculate_bert_score, calculate_llm_similarity
from utils import load_chunks, process_json_batch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from app.py to avoid duplication
from app import create_app, db, login_manager

# Create the app
app = create_app()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max-limit

# Import models after initializing db to avoid circular import
from models import User, Conversation, Message, History, MinistryContact

# Create database tables
with app.app_context():
    db.create_all()
    logger.info("Checking if database tables exist")

# Load labor law chunks
def get_chunks():
    """Return the labor law chunks for use in other modules"""
    from utils import load_chunks
    chunks = load_chunks()
    logger.info(f"Processed {len(chunks)} chunks with embeddings")
    return chunks

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    with app.app_context():
        return User.query.get(int(user_id))

# Import routes after creating app
try:
    from routes import *
    logger.info("Successfully imported routes")
except Exception as e:
    logger.error(f"Error importing routes: {str(e)}")

# Basic routes that should stay in main.py
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/chat')
def chat():
    """Render the chat interface"""
    return render_template('index.html')

@app.route('/chat_simplified')
def chat_simplified():
    """Render the simplified chat interface"""
    return render_template('chat_simplified.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember_me = 'remember_me' in request.form
        
        if not username or not password:
            flash('يرجى إدخال اسم المستخدم وكلمة المرور', 'danger')
            return redirect(url_for('login'))
        
        try:
            # Find user - using app_context to ensure proper database access
            with app.app_context():
                user = User.query.filter_by(username=username).first()
                
                # Check credentials
                if user and check_password_hash(user.password_hash, password):
                    login_user(user, remember=remember_me)
                    next_page = request.args.get('next')
                    if not next_page or not next_page.startswith('/'):
                        next_page = url_for('index')
                    return redirect(next_page)
                else:
                    flash('اسم المستخدم أو كلمة المرور غير صحيحة', 'danger')
                    return redirect(url_for('login'))
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            flash('حدث خطأ أثناء تسجيل الدخول. يرجى المحاولة مرة أخرى.', 'danger')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate form data
        if not username or not password:
            flash('يرجى ملء جميع الحقول', 'danger')
            return redirect(url_for('register'))
        
        # Set default values for missing fields
        if not email:
            email = f"{username}@example.com"  # Generate a default email
            
        # For backward compatibility, set confirm_password if not provided
        if not confirm_password:
            confirm_password = password
            
        if password != confirm_password:
            flash('كلمات المرور غير متطابقة', 'danger')
            return redirect(url_for('register'))
        
        # Check for existing user
        existing_user = User.query.filter(User.username == username).first()
        if existing_user:
            flash('اسم المستخدم مستخدم بالفعل', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        new_user = User()
        new_user.username = username
        # No email field in our User model
        new_user.set_password(password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('تم إنشاء الحساب بنجاح! يمكنك الآن تسجيل الدخول', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating user: {str(e)}")
            flash('حدث خطأ أثناء إنشاء الحساب. يرجى المحاولة مرة أخرى', 'danger')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    """Handle user logout"""
    logout_user()
    flash('تم تسجيل الخروج بنجاح', 'success')
    return redirect(url_for('index'))

@app.route('/conversations', methods=['GET'])
@app.route('/api/conversations', methods=['GET'])  # Add API route for compatibility
@login_required
def get_conversations():
    """Get user's conversations"""
    conversations = Conversation.query.filter_by(user_id=current_user.id).order_by(Conversation.updated_at.desc()).all()
    return jsonify({
        'conversations': [
            {
                'id': conv.id, 
                'title': conv.title,
                'created_at': conv.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'updated_at': conv.updated_at.strftime('%Y-%m-%d %H:%M:%S') if conv.updated_at else None
            } for conv in conversations
        ]
    })

@app.route('/conversations', methods=['POST'])
@app.route('/api/conversations', methods=['POST'])  # Add API route for compatibility
@login_required
def create_conversation():
    """Create a new conversation"""
    data = request.json
    title = data.get('title', 'محادثة جديدة')
    
    conversation = Conversation()
    conversation.title = title
    conversation.user_id = current_user.id
    
    db.session.add(conversation)
    db.session.commit()
    
    return jsonify({
        'id': conversation.id,
        'title': conversation.title,
        'created_at': conversation.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        'updated_at': conversation.updated_at.strftime('%Y-%m-%d %H:%M:%S') if conversation.updated_at else None
    })

@app.route('/conversations/<int:conversation_id>', methods=['GET'])
@app.route('/api/conversations/<int:conversation_id>', methods=['GET'])  # Add API route for compatibility
@login_required
def get_conversation(conversation_id):
    """Get a specific conversation with its messages"""
    conversation = Conversation.query.filter_by(id=conversation_id, user_id=current_user.id).first()
    
    if not conversation:
        return jsonify({'error': 'المحادثة غير موجودة'}), 404
    
    messages = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.timestamp).all()
    
    return jsonify({
        'conversation': {
            'id': conversation.id,
            'title': conversation.title,
            'created_at': conversation.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': conversation.updated_at.strftime('%Y-%m-%d %H:%M:%S') if conversation.updated_at else None
        },
        'messages': [
            {
                'id': msg.id,
                'content': msg.content,
                'is_user': msg.is_user,
                'created_at': msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            } for msg in messages
        ]
    })

@app.route('/conversations/<int:conversation_id>', methods=['PUT'])
@app.route('/api/conversations/<int:conversation_id>', methods=['PUT'])  # Add API route for compatibility
@login_required
def update_conversation(conversation_id):
    """Update conversation title"""
    conversation = Conversation.query.filter_by(id=conversation_id, user_id=current_user.id).first()
    
    if not conversation:
        return jsonify({'error': 'المحادثة غير موجودة'}), 404
    
    data = request.json
    if 'title' in data:
        conversation.title = data['title']
        db.session.commit()
    
    return jsonify({
        'id': conversation.id,
        'title': conversation.title,
        'created_at': conversation.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        'updated_at': conversation.updated_at.strftime('%Y-%m-%d %H:%M:%S') if conversation.updated_at else None
    })

@app.route('/conversations/<int:conversation_id>', methods=['DELETE'])
@app.route('/api/conversations/<int:conversation_id>', methods=['DELETE'])  # Add API route for compatibility
@login_required
def delete_conversation(conversation_id):
    """Delete a conversation"""
    conversation = Conversation.query.filter_by(id=conversation_id, user_id=current_user.id).first()
    
    if not conversation:
        return jsonify({'error': 'المحادثة غير موجودة'}), 404
    
    # Delete all messages
    Message.query.filter_by(conversation_id=conversation_id).delete()
    
    # Delete conversation
    db.session.delete(conversation)
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/conversations/<int:conversation_id>/messages', methods=['POST'])
@app.route('/api/conversations/<int:conversation_id>/messages', methods=['POST'])  # Add API route for compatibility
@login_required
def send_message(conversation_id):
    """Send a message in a conversation"""
    data = request.json
    content = data.get('content')
    
    if not content:
        return jsonify({'error': 'الرجاء إدخال محتوى الرسالة'}), 400
    
    conversation = Conversation.query.filter_by(id=conversation_id, user_id=current_user.id).first()
    if not conversation:
        return jsonify({'error': 'المحادثة غير موجودة'}), 404
    
    # Create user message
    user_message = Message()
    user_message.content = content
    user_message.is_user = True
    user_message.conversation_id = conversation_id
    db.session.add(user_message)
    
    # Create system response
    system_message = Message()
    system_message.content = "عذراً، لا يمكنني الرد على الرسائل مباشرة. يرجى استخدام نموذج السؤال في الصفحة الرئيسية."
    system_message.is_user = False
    system_message.conversation_id = conversation_id
    db.session.add(system_message)
    
    # Update conversation timestamp
    conversation.updated_at = datetime.utcnow()
    
    db.session.commit()
    
    return jsonify({
        'status': 'success',
        'message': 'تم إرسال الرسالة بنجاح',
        'messages': [
            {
                'id': user_message.id,
                'content': user_message.content,
                'is_user': user_message.is_user,
                'timestamp': user_message.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'id': system_message.id,
                'content': system_message.content,
                'is_user': system_message.is_user,
                'timestamp': system_message.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            }
        ]
    })

@app.route('/history')
@login_required
def history():
    """Show user's question history"""
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    # Get paginated history entries
    history_entries = History.query.filter_by(user_id=current_user.id).order_by(History.timestamp.desc()).paginate(page=page, per_page=per_page)
    
    return render_template('history.html', history=history_entries)

@app.route('/history/<int:history_id>')
@login_required
def history_detail(history_id):
    """Show details of a specific history entry"""
    history_entry = History.query.filter_by(id=history_id, user_id=current_user.id).first_or_404()
    
    # Log the attributes available on the History entry
    logger.info(f"History entry attributes: {dir(history_entry)}")
    logger.info(f"History entry data: id={history_entry.id}, query={history_entry.user_query}")
    
    # Parse referenced_chunks if available
    chunks = []
    # Safe check for attribute existence before accessing
    if hasattr(history_entry, 'referenced_chunks') and history_entry.referenced_chunks:
        try:
            chunks = json.loads(history_entry.referenced_chunks)
            logger.info(f"Successfully parsed referenced_chunks into {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error parsing referenced_chunks: {e}")
            chunks = []
    else:
        logger.warning(f"No referenced_chunks attribute or empty value for history_id={history_id}")
    
    return render_template('history_detail.html', entry=history_entry, chunks=chunks)

@app.route('/profile')
@login_required
def profile():
    """Show user profile"""
    question_count = History.query.filter_by(user_id=current_user.id).count()
    conversation_count = Conversation.query.filter_by(user_id=current_user.id).count()
    
    return render_template('profile.html', 
                           question_count=question_count, 
                           conversation_count=conversation_count)

@app.route('/contacts')
def contacts():
    """Show ministry contacts"""
    if not MinistryContact.query.first():
        # Add sample contacts if none exist
        contacts = [
            MinistryContact(
                name="مكتب وزير العمل",
                phone="065802666",
                email="info@mol.gov.jo",
                address="عمان - الشميساني - شارع العرموطي - وزارة العمل"
            ),
            MinistryContact(
                name="مركز الاتصال الوطني",
                phone="065008080",
                email="complaints@mol.gov.jo",
                address="عمان - العبدلي - وزارة العمل"
            ),
            MinistryContact(
                name="مديرية تفتيش العمل",
                phone="0799000000",
                email="inspection@mol.gov.jo",
                address="عمان - الشميساني - شارع العرموطي - وزارة العمل"
            )
        ]
        db.session.add_all(contacts)
        db.session.commit()
    
    # Get all contacts from database
    contacts = MinistryContact.query.all()
    
    return render_template('contacts.html', contacts=contacts)

@app.route('/evaluate_text', methods=['GET', 'POST'])
@login_required
def evaluate_text():
    """Simple text-based evaluation without file upload"""
    # Check if user is authenticated
    if not current_user.is_authenticated:
        flash('يرجى تسجيل الدخول للوصول إلى صفحة تقييم النظام', 'warning')
        return redirect(url_for('login', next=request.url))
    
    # Only allow superuser (test) to access evaluation page
    if not (current_user.username == 'test' or current_user.id == 1):
        flash('عذراً، صفحة تقييم النظام متاحة فقط للمشرفين', 'danger')
        return redirect(url_for('index'))
    
    results = None
    metrics = None
    
    if request.method == 'POST':
        # Get questions and answers as text blocks
        questions_text = request.form.get('questions_text', '')
        answers_text = request.form.get('answers_text', '')
        
        # Split text by lines and strip whitespace
        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
        answers = [a.strip() for a in answers_text.split('\n') if a.strip()]
        
        # Ensure equal number of questions and answers
        min_len = min(len(questions), len(answers))
        if min_len == 0:
            flash('يرجى إدخال على الأقل سؤال واحد وإجابة واحدة', 'danger')
            return render_template('evaluate_text.html')
        
        questions = questions[:min_len]
        answers = answers[:min_len]
        
        # Initialize agent chain with chunks from utils
        from utils import load_chunks
        chunks = load_chunks()
        from embeddings import EmbeddingProcessor
        embedding_processor = EmbeddingProcessor()
        from enhanced_agents import EnhancedAgentChain
        agent_chain = EnhancedAgentChain(chunks, embedding_processor)
        
        # Define error indicators
        ERROR_INDICATORS = [
            "حدث خطأ أثناء معالجة السؤال",
            "خطأ في معالجة الطلب",
            "حدث خطأ في النظام",
            "You exceeded your current quota", 
            "429",
            "exceeded rate limits"
        ]
        
        def is_error_response(text):
            return any(indicator in text for indicator in ERROR_INDICATORS)
        
        results = []
        all_bleu_scores = []
        all_bert_scores = []
        all_llm_scores = []
        
        # Process in small batches to prevent memory issues
        batch_size = 3  # Process 3 questions at a time
        num_batches = (min_len + batch_size - 1) // batch_size  # Ceiling division
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, min_len)
            logger.info(f"Processing batch {batch_idx+1}/{num_batches} (questions {start_idx+1}-{end_idx})")
            
            batch_questions = questions[start_idx:end_idx]
            batch_answers = answers[start_idx:end_idx]
            
            for i, (question, expected) in enumerate(zip(batch_questions, batch_answers)):
                global_idx = start_idx + i
                logger.info(f"Processing question {global_idx+1}/{min_len}: {question[:50]}...")
                
                # Try up to 3 times with increasing delays
                max_retries = 3
                retry_count = 0
                prediction = None
                
                while retry_count < max_retries:
                    try:
                        prediction, used_chunks = agent_chain.process_question(question)
                        
                        if is_error_response(prediction):
                            logger.warning(f"Error response for question {global_idx+1}, retrying...")
                            retry_count += 1
                            time.sleep(1 * (2 ** retry_count))
                            continue
                        else:
                            break
                    except Exception as e:
                        logger.error(f"Error processing question {global_idx+1}: {str(e)}")
                        retry_count += 1
                        if retry_count < max_retries:
                            time.sleep(1 * (2 ** retry_count))
                        else:
                            prediction = "عذراً، حدث خطأ أثناء معالجة هذا السؤال."
                            break
                
                # Calculate scores
                try:
                    bleu = calculate_bleu_score(prediction, expected)
                except Exception as e:
                    logger.error(f"BLEU score error: {str(e)}")
                    bleu = 0
                
                try:
                    bert = calculate_bert_score(prediction, expected)
                except Exception as e:
                    logger.error(f"BERT score error: {str(e)}")
                    bert = 0
                
                try:
                    llm = calculate_llm_similarity(prediction, expected)
                except Exception as e:
                    logger.error(f"LLM similarity error: {str(e)}")
                    llm = 0.5
                
                all_bleu_scores.append(bleu)
                all_bert_scores.append(bert)
                all_llm_scores.append(llm)
                
                results.append({
                    'id': global_idx + 1,
                    'question': question,
                    'ground_truth': expected,
                    'prediction': prediction,
                    'bleu_score': bleu,
                    'bert_score': bert,
                    'llm_score': llm
                })
            
            # Force garbage collection between batches to free memory
            gc.collect()
        
        # Calculate metrics
        metrics = {
            'total_questions': len(results),
            'average_bleu_score': sum(all_bleu_scores) / len(all_bleu_scores) if all_bleu_scores else 0,
            'average_bert_score': sum(all_bert_scores) / len(all_bert_scores) if all_bert_scores else 0,
            'average_llm_score': sum(all_llm_scores) / len(all_llm_scores) if all_llm_scores else 0,
            'bleu_scores': all_bleu_scores,
            'bert_scores': all_bert_scores,
            'llm_scores': all_llm_scores
        }
        
        flash(f'تم تقييم {len(results)} من الأسئلة بنجاح', 'success')
    
    return render_template('evaluate_text.html', results=results, metrics=metrics)


@app.route('/evaluate', methods=['GET'])
@login_required
def evaluate():
    """Evaluate the performance of the system with test data"""
    # Check if user is authenticated
    if not current_user.is_authenticated:
        flash('يرجى تسجيل الدخول للوصول إلى صفحة تقييم النظام', 'warning')
        return redirect(url_for('login', next=request.url))
    
    # Only allow superuser (test) to access evaluation page
    if not (current_user.username == 'test' or current_user.id == 1):
        flash('عذراً، صفحة تقييم النظام متاحة فقط للمشرفين', 'danger')
        return redirect(url_for('index'))
    
    
    # GET request (or POST with errors)
    return render_template('evaluate.html')
    
@app.route('/specific_test', methods=['GET'])
@login_required
def specific_test():
    """Process and evaluate the specific test file in attached_assets"""
    # Check if user is authenticated
    if not current_user.is_authenticated:
        flash('يرجى تسجيل الدخول للوصول إلى صفحة تقييم النظام', 'warning')
        return redirect(url_for('login', next=request.url))
    
    # Only allow superuser (test) to access evaluation page
    if not (current_user.username == 'test' or current_user.id == 1):
        flash('عذراً، صفحة تقييم النظام متاحة فقط للمشرفين', 'danger')
        return redirect(url_for('index'))
    
    # Import the processor function
    from process_specific_file import process_specific_json
    
    # Process the specific file
    metrics, results = process_specific_json()
    
    # Check if error occurred
    if isinstance(metrics, dict) and "error" in metrics:
        error = metrics["error"]
        flash(f'حدث خطأ أثناء معالجة الملف: {error}', 'danger')
        return render_template('specific_test_results.html', error=error)
    
    flash(f'تم تقييم {metrics["total_questions"]} من الأسئلة بنجاح', 'success')
    return render_template('specific_test_results.html', metrics=metrics, results=results)

@app.route('/test_attached_file', methods=['GET'])
def test_attached_file():
    """Test the specific JSON file from attached_assets folder"""
    try:
        # Run the process_specific_json function from process_specific_file.py
        from process_specific_file import process_specific_json
        metrics, results = process_specific_json()
        
        if isinstance(metrics, dict) and "error" in metrics:
            return render_template('error.html', error=metrics["error"])
        
        return render_template('evaluation_results.html', results=results, metrics=metrics)
    except Exception as e:
        app.logger.error(f"Error testing attached file: {str(e)}")
        return render_template('error.html', error=f"Error testing attached file: {str(e)}")


@app.route('/direct_json_test', methods=['GET'])
def direct_json_test():
    """
    Show page with button to start evaluation or start processing directly if auto=true.
    Uses the dedicated handler for this specific format.
    """
    # Check if auto parameter is present for automatic start
    auto_start = request.args.get('auto', 'false').lower() == 'true'
    
    if auto_start:
        # Auto-start the evaluation immediately
        return start_direct_evaluation()
    
    # Otherwise show the page with the button
    return render_template('evaluation_start.html')

@app.route('/start_direct_evaluation', methods=['GET', 'POST'])
def start_direct_evaluation():
    """
    Start processing the Arabic JSON file in the background and redirect to loading page.
    """
    try:
        # Import and use the dedicated processor
        from direct_json_handler import start_background_processing
        
        # Start processing in the background
        logger.info("Starting background processing with direct JSON handler")
        process_id = start_background_processing()
        
        # Check if we got a valid process ID
        if not process_id:
            logger.error("Failed to get process ID from start_background_processing")
            flash("حدث خطأ أثناء بدء المعالجة، يرجى المحاولة مرة أخرى", 'danger')
            return redirect(url_for('direct_json_test'))
        
        logger.info(f"Started direct JSON test with process ID: {process_id}")
        
        # Redirect to loading page with process ID
        return redirect(url_for('loading_page', process_id=process_id))
        
    except Exception as e:
        import traceback
        logger.error(f"Error in start_direct_evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f"حدث خطأ أثناء المعالجة: {str(e)}", 'danger')
        return redirect(url_for('direct_json_test'))

@app.route('/loading', methods=['GET'])
def loading_page():
    """Show loading page with process ID"""
    process_id = request.args.get('process_id')
    
    # If no process ID provided, redirect to the test page
    if not process_id:
        logger.warning("No process ID provided to loading page")
        return redirect(url_for('test_arabic_json'))
        
    # Check if the process exists
    from processing_manager import ProcessManager
    if not ProcessManager.get_process(process_id):
        logger.warning(f"Process ID not found: {process_id}")
        # Start a new process instead of showing an error
        from direct_json_handler import start_background_processing
        try:
            new_process_id = start_background_processing()
            logger.info(f"Started new process: {new_process_id}")
            return redirect(url_for('loading_page', process_id=new_process_id))
        except Exception as e:
            logger.error(f"Failed to start new process: {e}")
            return redirect(url_for('test_arabic_json'))
    
    return render_template('loading.html', process_id=process_id)

@app.route('/check-processing-status', methods=['GET'])
def check_processing_status():
    """API endpoint to check the status of file processing"""
    try:
        from processing_manager import ProcessManager
        
        # Get process ID from query parameters
        process_id = request.args.get('process_id')
        if not process_id:
            logger.warning("No process ID provided to check_processing_status")
            return jsonify({
                'status': 'redirect',
                'message': 'No process ID provided',
                'redirect_url': url_for('test_arabic_json')
            })
        
        # Get process info, now this will check disk storage if not in memory
        process_info = ProcessManager.get_process(process_id)
        if not process_info:
            logger.error(f"Process {process_id} not found in memory or disk")
            return jsonify({
                'status': 'error',
                'message': 'Process not found'
            })
        
        # Prepare response
        response = {
            'status': process_info.get('status'),
            'message': process_info.get('message'),
            'progress': process_info.get('progress', 0),
            'processed_questions': process_info.get('processed_items', 0),
            'total_questions': process_info.get('total_items', 0),
        }
        
        # Include partial results count if available
        partial_results = process_info.get('partial_results', [])
        if partial_results:
            response['partial_results_count'] = len(partial_results)
            
        # Add redirect URL if processing is complete or if we have partial results to show
        if process_info.get('status') == 'completed':
            response['redirect_url'] = f"/direct-json-results?process_id={process_id}"
        elif len(partial_results) > 0 and process_info.get('error'):
            # If we have an error but partial results are available, allow viewing them
            response['has_partial_results'] = True
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error checking process status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error checking status: {str(e)}'
        })

@app.route('/direct-json-results', methods=['GET'])
def direct_json_results():
    """
    Display results from background processing
    """
    try:
        from processing_manager import ProcessManager
        
        process_id = request.args.get('process_id')
        if not process_id:
            return render_template('error.html', 
                                  error="معرف العملية غير متوفر",
                                  details="تعذر العثور على معرف العملية في طلب URL.")
        
        # Get process info - uses the enhanced version that checks disk storage
        process_info = ProcessManager.get_process(process_id)
        if not process_info:
            logger.error(f"Process {process_id} not found in memory or disk")
            return render_template('error.html', 
                                  error="معلومات العملية غير متوفرة",
                                  details="تعذر العثور على معلومات العملية للمعرف المحدد.")
        
        # Get results or partial results
        results = process_info.get('results', [])
        
        # If no results but we have partial results, use those
        if not results and process_info.get('partial_results'):
            results = process_info.get('partial_results', [])
            # Add a flag to indicate these are partial results
            using_partial_results = True
        else:
            using_partial_results = False
            
        # Check if processing is complete or if we have partial results
        if process_info.get('status') != 'completed' and not using_partial_results:
            # Redirect back to loading page
            return render_template('loading.html', process_id=process_id)
        
        file_info = process_info.get('file_info', {})
        
        # Get metrics if available
        metrics = file_info.get('metrics', {})
        
        # If we're using partial results, calculate metrics for them
        if using_partial_results and results:
            # Collect scores, filtering out None values
            bleu_scores = [r.get("bleu_score", 0) for r in results if r.get("bleu_score") is not None]
            bert_scores = [r.get("bert_score", 0) for r in results if r.get("bert_score") is not None]
            llm_scores = [r.get("llm_score", 0) for r in results if r.get("llm_score") is not None]
            
            if bleu_scores:
                metrics["average_bleu_score"] = sum(bleu_scores) / len(bleu_scores)
            
            if bert_scores:
                metrics["average_bert_score"] = sum(bert_scores) / len(bert_scores)
            
            if llm_scores:
                metrics["average_llm_score"] = sum(llm_scores) / len(llm_scores)
        
        # Add a note about partial results if applicable
        title = f"نتائج اختبار ملف: {file_info.get('filename', 'JSON')}"
        if using_partial_results:
            title += f" (نتائج جزئية - اكتملت حتى السؤال {len(results)})"
            
        return render_template('direct_json_results.html',
                              results=results,
                              title=title,
                              total_questions=file_info.get('total_questions', len(results)),
                              processed_questions=len(results),
                              process_id=process_id,
                              metrics=metrics,
                              average_bleu_score=metrics.get('average_bleu_score', 0),
                              average_bert_score=metrics.get('average_bert_score', 0),
                              average_llm_score=metrics.get('average_llm_score', 0),
                              is_partial=using_partial_results,
                              error_message=process_info.get('error') if using_partial_results else None)
                              
    except Exception as e:
        import traceback
        logger.error(f"Error displaying results: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('error.html',
                             error=f"حدث خطأ أثناء عرض النتائج: {str(e)}",
                             details=traceback.format_exc())


@app.route('/test-arabic-json', methods=['GET', 'POST'])
def test_arabic_json():
    """Test loading and processing Arabic JSON files with a user-friendly interface"""
    # Display the form for GET requests
    if request.method == 'GET':
        return render_template('arabic_json_form.html')
        
    # Process form submission for POST requests
    try:
        from enhanced_agents import EnhancedAgentChain
        from embeddings import EmbeddingProcessor
        from evaluation import calculate_bleu_score, calculate_bert_score, calculate_llm_similarity
            
        # Initialize agent chain
        chunks = get_chunks()
        embedding_processor = EmbeddingProcessor()
        agent_chain = EnhancedAgentChain(chunks, embedding_processor)
        
        # Get the method from the form
        method = request.form.get('method')
        file_content = None
        loaded_file = None
        used_encoding = None
        source_info = "Direct input"
        
        # Process based on the selected method
        if method == 'built_in':
            # Use built-in file from attached_assets
            potential_filenames = [
                "تنسيق_القوائم222_(أسماء_حقول_بديلة)[1].json",
                "تنسيق_القوائم222_(أسماء_حقول_بديلة).json",
                "تنسيق_القوائم222.json"
            ]
            
            for filename in potential_filenames:
                filepath = os.path.join("attached_assets", filename)
                if os.path.exists(filepath):
                    # Try different encodings
                    encodings = ['utf-8', 'utf-8-sig', 'cp1256', 'latin-1']
                    for encoding in encodings:
                        try:
                            with open(filepath, 'r', encoding=encoding) as f:
                                file_content = f.read()
                                # Test if encoding worked
                                json.loads('{"test": "' + 'تجربة' + '"}')
                                loaded_file = filepath
                                used_encoding = encoding
                                source_info = f"Built-in file: {os.path.basename(filepath)}"
                                logger.info(f"Successfully decoded file {filename} with {encoding} encoding")
                                break
                        except Exception as e:
                            logger.warning(f"Failed to decode file {filename} with {encoding}: {str(e)}")
                            continue
                    
                    if loaded_file:
                        break  # Successfully loaded a file
            
            if not loaded_file:
                return render_template('error.html', error='لم يتم العثور على ملف JSON عربي أو تعذر قراءته')
                
        elif method == 'upload':
            # Process uploaded file
            if 'file' not in request.files:
                return render_template('error.html', error='لم يتم تحميل أي ملف')
                
            file = request.files['file']
            if not file or file.filename == '':
                return render_template('error.html', error='لم يتم اختيار ملف')
            
            if not file.filename or not file.filename.lower().endswith('.json'):
                return render_template('error.html', error='يجب أن يكون الملف بتنسيق JSON')
            
            # Read the file with different encodings
            encodings = ['utf-8', 'utf-8-sig', 'cp1256', 'latin-1']
            for encoding in encodings:
                try:
                    file.seek(0)  # Reset file position
                    file_content = file.read().decode(encoding)
                    loaded_file = file.filename
                    used_encoding = encoding
                    source_info = f"Uploaded file: {file.filename}"
                    logger.info(f"Successfully decoded uploaded file with {encoding} encoding")
                    break
                except Exception as e:
                    logger.warning(f"Failed to decode uploaded file with {encoding}: {str(e)}")
                    continue
            
            if file_content is None:
                return render_template('error.html', error='تعذر قراءة محتوى الملف بأي ترميز معروف')
                
        elif method == 'paste':
            # Process pasted JSON content
            file_content = request.form.get('json_content')
            if not file_content:
                return render_template('error.html', error='لم يتم إدخال أي محتوى JSON')
            
            loaded_file = "pasted_content"
            used_encoding = "utf-8"
            source_info = "Pasted JSON content"
            
        else:
            return render_template('error.html', error='طريقة غير صالحة')
        
        # Process the JSON content
        try:
            if file_content is None:
                raise ValueError("File content is empty")
                
            data = json.loads(file_content)
            
            # Extract questions and answers
            questions = []
            expected_answers = []
            
            # Handle different JSON formats
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Check for different possible key names for questions
                        question = None
                        answer = None
                        
                        # Check for different possible key names for questions
                        for q_key in ['question', 'q', 'سؤال', 'س']:
                            if q_key in item:
                                question = item[q_key]
                                break
                        
                        # Check for different possible key names for answers
                        for a_key in ['answer', 'a', 'جواب', 'إجابة', 'ج']:
                            if a_key in item:
                                answer = item[a_key]
                                break
                        
                        if question is not None:
                            questions.append(question)
                            expected_answers.append(answer if answer is not None else "")
            else:
                raise ValueError("بنية JSON غير صالحة. يجب أن تكون البيانات على شكل مصفوفة من الكائنات.")
            
            # Check if we have questions
            if not questions:
                return render_template('error.html', error='لم يتم العثور على أي أسئلة في البيانات المقدمة')
            
            # Process each question sequentially
            results = []
            all_bleu_scores = []
            all_bert_scores = []
            all_llm_scores = []
            
            for i, (question, expected) in enumerate(zip(questions, expected_answers)):
                try:
                    # Add a small delay to avoid rate limiting
                    if i > 0:
                        import time
                        import random
                        delay = 1.0 + random.random()
                        logger.info(f"Waiting {delay:.2f} seconds before processing next question...")
                        time.sleep(delay)
                        
                    # Process the question
                    logger.info(f"Processing question {i+1}/{len(questions)}: {question}")
                    generated, used_chunks = agent_chain.process_question(question)
                    
                    # Calculate metrics
                    try:
                        bleu = calculate_bleu_score(generated, expected)
                    except Exception as e:
                        logger.error(f"Error calculating BLEU score: {str(e)}")
                        bleu = 0
                        
                    try:
                        bert = calculate_bert_score(generated, expected)
                    except Exception as e:
                        logger.error(f"Error calculating BERT score: {str(e)}")
                        bert = 0
                        
                    try:
                        llm = calculate_llm_similarity(generated, expected)
                    except Exception as e:
                        logger.error(f"Error calculating LLM similarity: {str(e)}")
                        llm = 0.5
                    
                    # Store results
                    result = {
                        'id': i + 1,
                        'question': question,
                        'ground_truth': expected,
                        'prediction': generated,
                        'bleu_score': bleu,
                        'bert_score': bert,
                        'llm_score': llm
                    }
                    
                    results.append(result)
                    all_bleu_scores.append(bleu)
                    all_bert_scores.append(bert)
                    all_llm_scores.append(llm)
                    
                except Exception as e:
                    logger.error(f"Error processing question {i+1}: {str(e)}")
                    logger.exception("Exception details:")
                    # Add a placeholder result
                    results.append({
                        'id': i + 1,
                        'question': question,
                        'ground_truth': expected,
                        'prediction': f"حدث خطأ أثناء معالجة هذا السؤال: {str(e)}",
                        'bleu_score': 0,
                        'bert_score': 0,
                        'llm_score': 0
                    })
            
            # Calculate overall metrics
            avg_bleu = sum(all_bleu_scores) / max(len(all_bleu_scores), 1)
            avg_bert = sum(all_bert_scores) / max(len(all_bert_scores), 1)
            avg_llm = sum(all_llm_scores) / max(len(all_llm_scores), 1)
            
            # Set up metrics
            metrics = {
                'total_questions': len(questions),
                'average_bleu_score': avg_bleu,
                'average_bert_score': avg_bert,
                'average_llm_score': avg_llm,
                'bleu_scores': all_bleu_scores,
                'bert_scores': all_bert_scores,
                'llm_scores': all_llm_scores,
                'file_info': {
                    'source': source_info,
                    'encoding': used_encoding,
                    'size': len(file_content) if file_content else 0,
                    'questions': len(questions)
                }
            }
            
            # Return evaluation results
            return render_template('evaluation_results.html', results=results, metrics=metrics)
            
        except Exception as e:
            import traceback
            logger.error(f"Error processing JSON content: {str(e)}")
            logger.exception("Exception details:")
            return render_template('error.html', 
                                 error=f"خطأ في معالجة محتوى JSON: {str(e)}",
                                 details=traceback.format_exc())
    except Exception as e:
        import traceback
        logger.error(f"Error in test_arabic_json: {str(e)}")
        logger.exception("Exception details:")
        return render_template('error.html', 
                             error=f"خطأ في اختبار ملف JSON العربي: {str(e)}",
                             details=traceback.format_exc())

@app.route('/static/download/example_json', methods=['GET'])
@login_required
def download_example_json():
    """Download an example JSON file for evaluation"""
    # Check if user is authenticated and is admin
    if not current_user.is_authenticated or not (current_user.username == 'test' or current_user.id == 1):
        flash('عذراً، هذه الصفحة متاحة فقط للمشرفين', 'danger')
        return redirect(url_for('index'))
        
    # Create example JSON data in the expected format
    example_data = [
        {
            "q": "ما هي مدة الإجازة السنوية التي يستحقها الموظف؟",
            "a": "يستحق العامل إجازة سنوية مدتها 14 يوم عمل للعامل الذي أمضى في الخدمة أقل من 5 سنوات و 21 يوم عمل للعامل الذي خدم 5 سنوات أو أكثر. المادة 61 من قانون العمل."
        },
        {
            "q": "هل يجوز فصل العامل بدون إشعار؟",
            "a": "لا يجوز فصل العامل دون توجيه إشعار خطي قبل شهر من تاريخ إنهاء العمل حسب المادة 23 من قانون العمل الأردني."
        },
        {
            "q": "ما هي حقوق المرأة الحامل في قانون العمل؟",
            "a": "للمرأة الحامل الحق في إجازة أمومة مدتها 10 أسابيع براتب كامل، ويجب منحها فترات رضاعة لا تقل عن ساعة يومياً. المادة 67 و 71 من قانون العمل."
        },
        {
            "q": "ما هو الحد الأدنى للأجور في الأردن؟",
            "a": "الحد الأدنى للأجور في الأردن هو 260 دينار أردني شهرياً وفقاً لآخر قرار للجنة الثلاثية لشؤون العمل."
        },
        {
            "q": "هل يحق للعامل الحصول على بدل تعويض عن العمل الإضافي؟",
            "a": "نعم، يحق للعامل الحصول على بدل عن ساعات العمل الإضافية بما لا يقل عن 125% من أجره العادي في أيام العمل العادية و150% في أيام العطل والأعياد والعطل الرسمية. المادة 59 من قانون العمل."
        }
    ]
    
    # Create a response with JSON data
    response = Response(
        json.dumps(example_data, ensure_ascii=False, indent=2),
        status=200,
        mimetype='application/json'
    )
    
    # Set headers to indicate this is a downloadable file
    response.headers["Content-Disposition"] = "attachment; filename=example_test_data.json"
    return response


@app.route('/download-results-csv/<process_id>', methods=['GET'])
def download_results_csv(process_id):
    """Download results as CSV file"""
    try:
        import csv
        import io
        from flask import Response
        from processing_manager import ProcessManager
        
        # Get process info
        process_info = ProcessManager.get_process(process_id)
        if not process_info:
            flash('لم يتم العثور على معلومات المعالجة', 'warning')
            return redirect(url_for('test_arabic_json'))
            
        # Get results and file info
        results = process_info.get('results', [])
        file_info = process_info.get('file_info', {})
        filename = file_info.get('filename', 'results')
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header row
        writer.writerow(['السؤال', 'الإجابة المتوقعة', 'إجابة النظام', 'درجة BLEU', 'درجة BERT', 'درجة LLM'])
        
        # Write result rows
        for result in results:
            writer.writerow([
                result.get('question', ''), 
                result.get('ground_truth', ''),
                result.get('prediction', ''),
                result.get('bleu_score', 0),
                result.get('bert_score', 0),
                result.get('llm_score', 0)
            ])
        
        # Create response with CSV file
        output.seek(0)
        response = Response(output.getvalue(), mimetype='text/csv')
        
        # Use a simple ASCII filename to avoid encoding issues with proper RFC format
        safe_filename = "evaluation_results.csv"
        response.headers["Content-Disposition"] = f'attachment; filename="{safe_filename}"'
        return response
        
    except Exception as e:
        import traceback
        logger.error(f"Error generating CSV: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f'حدث خطأ أثناء إنشاء ملف CSV: {str(e)}', 'danger')
        return redirect(url_for('direct_json_results', process_id=process_id))
        
@app.route('/static/download/example_csv', methods=['GET'])
@login_required
def download_example_csv():
    """Download an example CSV file for evaluation"""
    # Check if user is authenticated and is admin
    if not current_user.is_authenticated or not (current_user.username == 'test' or current_user.id == 1):
        flash('عذراً، هذه الصفحة متاحة فقط للمشرفين', 'danger')
        return redirect(url_for('index'))
        
    # Create CSV content
    csv_content = "question,answer\n"
    csv_content += "ما هي مدة الإجازة السنوية التي يستحقها الموظف؟,\"يستحق العامل إجازة سنوية مدتها 14 يوم عمل للعامل الذي أمضى في الخدمة أقل من 5 سنوات و 21 يوم عمل للعامل الذي خدم 5 سنوات أو أكثر. المادة 61 من قانون العمل.\"\n"
    csv_content += "هل يجوز فصل العامل بدون إشعار؟,\"لا يجوز فصل العامل دون توجيه إشعار خطي قبل شهر من تاريخ إنهاء العمل حسب المادة 23 من قانون العمل الأردني.\"\n"
    csv_content += "ما هي حقوق المرأة الحامل في قانون العمل؟,\"للمرأة الحامل الحق في إجازة أمومة مدتها 10 أسابيع براتب كامل، ويجب منحها فترات رضاعة لا تقل عن ساعة يومياً. المادة 67 و 71 من قانون العمل.\"\n"
    csv_content += "ما هو الحد الأدنى للأجور في الأردن؟,\"الحد الأدنى للأجور في الأردن هو 260 دينار أردني شهرياً وفقاً لآخر قرار للجنة الثلاثية لشؤون العمل.\"\n"
    csv_content += "هل يحق للعامل الحصول على بدل تعويض عن العمل الإضافي؟,\"نعم، يحق للعامل الحصول على بدل عن ساعات العمل الإضافية بما لا يقل عن 125% من أجره العادي في أيام العمل العادية و150% في أيام العطل والأعياد والعطل الرسمية. المادة 59 من قانون العمل.\"\n"
    
    # Create a response with CSV data
    response = Response(
        csv_content,
        status=200,
        mimetype='text/csv'
    )
    
    # Set headers to indicate this is a downloadable file
    response.headers["Content-Disposition"] = "attachment; filename=example_test_data.csv"
    return response


@app.route('/api/evaluate/direct', methods=['POST'])
@login_required
def evaluate_direct():
    """Handle direct JSON data upload through API endpoint"""
    # Check if user is authenticated and is admin
    if not current_user.is_authenticated or not (current_user.username == 'test' or current_user.id == 1):
        return jsonify({'error': 'غير مصرح لك بالوصول إلى هذه الصفحة'}), 403
    
    # Get JSON data from request
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({'error': 'لم يتم استلام بيانات JSON صالحة'}), 400
        
        # Log information about the received data
        logger.info(f"Received direct JSON data: {type(data)}")
        
        # Handle both array and single item formats
        if isinstance(data, list):
            test_data = data
        else:
            test_data = [data]
        
        # Process the test data
        questions = []
        expected_answers = []
        
        for item in test_data:
            if isinstance(item, dict):
                # Support various field naming conventions
                if 'q' in item and 'a' in item:
                    questions.append(item['q'])
                    expected_answers.append(item['a'])
                elif 'question' in item and 'answer' in item:
                    questions.append(item['question'])
                    expected_answers.append(item['answer'])
                elif 'input' in item and 'output' in item:
                    questions.append(item['input'])
                    expected_answers.append(item['output'])
                elif 'prompt' in item and 'expected' in item:
                    questions.append(item['prompt'])
                    expected_answers.append(item['expected'])
        
        if not questions:
            return jsonify({'error': 'لم يتم العثور على أسئلة صالحة في البيانات المقدمة'}), 400
        
        # Initialize agent chain
        agent_chain = EnhancedAgentChain(load_chunks())
        
        # Process questions and generate answers
        results = []
        total_bleu = 0
        total_bert = 0
        total_llm = 0
        
        for i, (question, expected) in enumerate(zip(questions, expected_answers)):
            # Process the question
            answer, used_chunks = agent_chain.process_question(question)
            
            # Calculate scores
            bleu = calculate_bleu_score(answer, expected)
            bert = calculate_bert_score(answer, expected)
            llm = calculate_llm_similarity(answer, expected)
            
            total_bleu += bleu
            total_bert += bert
            total_llm += llm
            
            # Add to results
            results.append({
                'id': i + 1,
                'question': question,
                'expected': expected,
                'answer': answer,
                'bleu': bleu,
                'bert': bert,
                'llm': llm
            })
        
        # Calculate averages
        num_questions = len(questions)
        avg_bleu = total_bleu / num_questions if num_questions > 0 else 0
        avg_bert = total_bert / num_questions if num_questions > 0 else 0
        avg_llm = total_llm / num_questions if num_questions > 0 else 0
        
        # Return results
        return jsonify({
            'status': 'success',
            'total_questions': num_questions,
            'average_bleu': avg_bleu,
            'average_bert': avg_bert,
            'average_llm': avg_llm,
            'results': results
        })
    
    except Exception as e:
        logger.error(f"Error processing direct JSON: {str(e)}")
        return jsonify({'error': f'حدث خطأ أثناء معالجة البيانات: {str(e)}'}), 500


@app.route('/api/evaluate/stream', methods=['POST'])
@login_required
def evaluate_stream():
    """Stream evaluation results as they are processed"""
    import time
    from flask import Response, stream_with_context
    import json
    from enhanced_agents import EnhancedAgentChain
    from embeddings import EmbeddingProcessor
    from utils import load_chunks, process_json_batch
    
    # Check if user is authenticated and is admin
    if not current_user.is_authenticated or not (current_user.username == 'test' or current_user.id == 1):
        return jsonify({'error': 'غير مصرح لك بالوصول إلى هذه الصفحة'}), 403
    
    # Process test data
    test_data = []
    data_received = False
    
    try:
        # Handle file upload
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            logger.info(f"DEBUG-UPLOAD: File received: {file.filename}")
            
            # Get file extension and mime type - safely handling None filenames
            filename = file.filename or ""
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Add additional check for content type
            content_type = file.content_type or ""
            logger.info(f"DEBUG-UPLOAD: File content type: {content_type}")
            
            # Initialize data variable
            data = None
            
            # Read the raw content first to better analyze the file
            raw_content = file.read()
            logger.info(f"DEBUG-UPLOAD: Read {len(raw_content)} bytes from file {filename}")
            
            # Save a copy of the original content for debugging
            with open("last_uploaded_file.bin", "wb") as debug_file:
                debug_file.write(raw_content)
            logger.info("DEBUG-UPLOAD: Saved a copy of the uploaded file for debugging")
            
            # Handle special character in filename (known issue with تنسيق_القوائم222 file)
            if 'تنسيق_القوائم' in filename:
                logger.info(f"DEBUG-UPLOAD: Special handling for file with known format issue: {filename}")
                try:
                    # Try to decode with different encodings
                    for encoding in ['utf-8', 'utf-8-sig', 'cp1256', 'latin-1']:
                        try:
                            content_str = raw_content.decode(encoding)
                            data = json.loads(content_str)
                            logger.info(f"DEBUG-UPLOAD: Successfully decoded special file with {encoding}")
                            
                            # Extract questions and answers
                            questions = []
                            expected_answers = []
                            
                            # Handle list format
                            if isinstance(data, list):
                                for item in data:
                                    if 'q' in item and 'a' in item:
                                        questions.append(item['q'])
                                        expected_answers.append(item['a'])
                            
                            if questions:
                                test_data = list(zip(questions, expected_answers))
                                data_received = True
                                logger.info(f"DEBUG-UPLOAD: Extracted {len(questions)} questions from special file")
                                break
                        except (UnicodeDecodeError, json.JSONDecodeError) as e:
                            logger.info(f"DEBUG-UPLOAD: Failed to decode with {encoding}: {str(e)}")
                            continue
                except Exception as e:
                    logger.error(f"Error processing special file: {str(e)}")
                    return jsonify({'error': f'خطأ في معالجة الملف الخاص: {str(e)}'}), 500
            
            if not data_received and (file_ext == '.json' or 'json' in content_type.lower() or (file_ext == '' and len(raw_content) > 0)):
                # Process JSON file
                try:
                    # Analyze first few bytes for JSON patterns
                    is_likely_json = False
                    first_chars = raw_content[:100]
                    try:
                        first_chars_str = first_chars.decode('utf-8', errors='replace').strip()
                        logger.info(f"DEBUG-UPLOAD: First characters: {first_chars_str[:50]}")
                        # Check for JSON start markers
                        if first_chars_str.startswith('[') or first_chars_str.startswith('{'):
                            is_likely_json = True
                            logger.info("DEBUG-UPLOAD: File starts with JSON markers ([ or {)")
                    except Exception as e:
                        logger.warning(f"DEBUG-UPLOAD: Error analyzing first bytes: {str(e)}")
                    
                    # Try multiple encodings for better compatibility
                    content = None
                    successful_encoding = None
                    
                    for encoding in ['utf-8-sig', 'utf-8', 'cp1256', 'latin-1']:
                        try:
                            content = raw_content.decode(encoding)
                            successful_encoding = encoding
                            logger.info(f"DEBUG-UPLOAD: Successfully decoded with {encoding}")
                            break
                        except UnicodeDecodeError:
                            logger.warning(f"DEBUG-UPLOAD: Failed to decode with {encoding}")
                            continue
                            
                    if content is None:
                        return jsonify({'error': 'فشل في قراءة الملف بأي من الترميزات المدعومة'}), 400
                    
                    # Check if content looks like HTML (starts with <!DOCTYPE or <html)
                    content_lower = content.strip().lower()
                    logger.info(f"JSON content starts with: {content_lower[:50]}")
                    
                    # Try to parse content directly first without HTML detection
                    try:
                        # If this succeeds, it's valid JSON, so we'll skip HTML detection
                        json.loads(content)
                        logger.info("Content is valid JSON, proceeding without HTML detection")
                    except json.JSONDecodeError:
                        # Only check for HTML if JSON parsing fails
                        logger.info("Initial JSON parse failed, checking if content is HTML")
                        
                        # Enhanced HTML detection with more patterns, but only looking at the beginning of the file
                        is_html = (
                            content_lower.startswith("<!doctype") or 
                            content_lower.startswith("<html") or
                            (content_lower.startswith("<") and "html" in content_lower[:20])
                        )
                        
                        if is_html:
                            logger.error(f"File appears to be HTML, not JSON. Content starts with: {content_lower[:50]}")
                            return jsonify({'error': 'الملف المرفوع يبدو أنه HTML وليس JSON. يرجى التأكد من تحميل ملف JSON صحيح.'}), 400
                    
                    # Try to parse content directly
                    try:
                        data = json.loads(content)
                        logger.info("Content successfully parsed as JSON")
                    except json.JSONDecodeError as e:
                        # If parsing fails, try to extract JSON patterns
                        logger.warning(f"Initial JSON parse failed, trying to extract JSON from content: {str(e)}")
                        
                        # Try several pattern matching approaches to find valid JSON
                        # 1. Look for array pattern with objects
                        json_pattern1 = re.compile(r'(\[[\s]*\{.*?\}[\s]*\])', re.DOTALL)
                        # 2. Look for object pattern
                        json_pattern2 = re.compile(r'(\{[\s]*".*?"[\s]*:.*?\})', re.DOTALL)
                        # 3. Look for anything between square brackets
                        json_pattern3 = re.compile(r'(\[.*?\])', re.DOTALL)
                        
                        # Try patterns in order of specificity
                        found_valid_json = False
                        for pattern, pattern_name in [
                            (json_pattern1, "array with objects"), 
                            (json_pattern2, "object pattern"),
                            (json_pattern3, "square brackets")
                        ]:
                            match = pattern.search(content)
                            if match:
                                potential_json = match.group(1)
                                logger.info(f"Found potential JSON with {pattern_name} pattern: {potential_json[:100]}...")
                                # Try parsing the extracted content
                                try:
                                    data = json.loads(potential_json)
                                    logger.info(f"Successfully extracted JSON with {pattern_name} pattern")
                                    found_valid_json = True
                                    break  # Exit loop if successful
                                except json.JSONDecodeError as e2:
                                    logger.warning(f"Failed to parse extracted JSON with {pattern_name} pattern: {str(e2)}")
                                    continue  # Try next pattern
                        
                        if not found_valid_json:
                            # This executes if no patterns worked
                            logger.error("Could not find valid JSON patterns in the content")
                            raise ValueError("No valid JSON pattern found in content")
                    
                    # Handle array format
                    if isinstance(data, list):
                        test_data = data
                        data_received = True
                    else:
                        return jsonify({'error': 'صيغة ملف JSON غير صحيحة. يجب أن يكون الملف على شكل مصفوفة من الأسئلة والأجوبة.'}), 400
                
                except Exception as e:
                    logger.error(f"Error processing JSON file: {str(e)}")
                    return jsonify({'error': f'خطأ في معالجة ملف JSON: {str(e)}'}), 400
                    
            elif file_ext == '.csv':
                # Process CSV file
                try:
                    import csv
                    import io
                    
                    # Already have raw_content from earlier
                    logger.info(f"Processing CSV file of {len(raw_content)} bytes")
                    
                    # Try different encodings for CSV file
                    encodings = ['utf-8-sig', 'utf-8', 'cp1256', 'latin-1']
                    content = None
                    successful_encoding = None
                    
                    for encoding in encodings:
                        try:
                            content = raw_content.decode(encoding)
                            successful_encoding = encoding
                            logger.info(f"DEBUG-UPLOAD: Successfully decoded CSV with {encoding}")
                            break
                        except UnicodeDecodeError:
                            logger.warning(f"DEBUG-UPLOAD: Failed to decode CSV with {encoding}")
                            continue
                    
                    if content is None:
                        return jsonify({'error': 'فشل في قراءة ملف CSV بأي من الترميزات المدعومة'}), 400
                    
                    # Use CSV reader to parse the content
                    file_like = io.StringIO(content)
                    reader = csv.reader(file_like)
                    
                    # Read header
                    header = next(reader, None)
                    if not header:
                        return jsonify({'error': 'ملف CSV فارغ أو غير صالح'}), 400
                    
                    # Determine column indices based on headers
                    question_idx = None
                    answer_idx = None
                    
                    for i, col in enumerate(header):
                        col_lower = col.lower().strip()
                        if col_lower in ['question', 'q', 'input', 'سؤال']:
                            question_idx = i
                        elif col_lower in ['answer', 'a', 'expected', 'output', 'جواب', 'إجابة']:
                            answer_idx = i
                    
                    # If standard headers not found, use first two columns
                    if question_idx is None or answer_idx is None:
                        question_idx = 0
                        answer_idx = 1
                        # Process from the beginning (including the header as first row)
                        file_like.seek(0)
                        reader = csv.reader(file_like)
                    
                    # Process rows
                    rows = []
                    for row in reader:
                        if len(row) > max(question_idx, answer_idx):
                            question = row[question_idx]
                            answer = row[answer_idx]
                            rows.append((question, answer))
                    
                    # Set test_data
                    if rows:
                        test_data = rows
                    
                    if test_data:
                        data_received = True
                    else:
                        return jsonify({'error': 'لم يتم العثور على بيانات صالحة في ملف CSV'}), 400
                        
                except Exception as e:
                    logger.error(f"Error processing CSV file: {str(e)}")
                    return jsonify({'error': f'خطأ في معالجة ملف CSV: {str(e)}'}), 400
            else:
                return jsonify({'error': 'نوع الملف غير مدعوم. يرجى استخدام ملفات JSON أو CSV فقط'}), 400
        
        # Handle direct JSON input
        elif request.form.get('json_input'):
            try:
                content = request.form.get('json_input')
                
                # Check if content is None or empty
                if not content:
                    logger.error("Empty JSON input received")
                    return jsonify({'error': 'لم يتم توفير أي مدخلات JSON'}), 400
                
                content_stripped = content.strip()
                if content_stripped == '':
                    logger.error("Empty JSON input (whitespace only) received")
                    return jsonify({'error': 'لم يتم توفير أي مدخلات JSON (فقط مسافات فارغة)'}), 400
                
                # Check if content looks like HTML (starts with <!DOCTYPE or <html)
                content_lower = content_stripped.lower()
                logger.info(f"JSON input starts with: {content_lower[:50]}")
                
                # Try to parse content directly first without HTML detection
                try:
                    # If this succeeds, it's valid JSON, so we'll skip HTML detection
                    json.loads(content_stripped)
                    logger.info("Content is valid JSON, proceeding without HTML detection")
                except json.JSONDecodeError:
                    # Only check for HTML if JSON parsing fails
                    logger.info("Initial JSON parse failed, checking if content is HTML")
                    
                    # Enhanced HTML detection with more patterns, but only looking at the beginning of the file
                    is_html = (
                        content_lower.startswith("<!doctype") or 
                        content_lower.startswith("<html") or
                        (content_lower.startswith("<") and "html" in content_lower[:20])
                    )
                    
                    if is_html:
                        logger.error(f"JSON input appears to be HTML, not JSON. Content starts with: {content_lower[:50]}")
                        return jsonify({'error': 'المدخلات تبدو أنها HTML وليست JSON. يرجى التأكد من إدخال JSON صحيح.'}), 400
                
                # Log preview of the content
                preview = content[:100] + "..." if len(content) > 100 else content
                logger.info(f"Parsing JSON input (preview): {preview}")
                
                # Try to extract JSON if content might be HTML with embedded JSON
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    # If parsing fails and content looks like it could be HTML, try to extract JSON
                    logger.warning(f"Initial JSON parse failed, trying to extract JSON from possible HTML: {str(e)}")
                    
                    # Try several pattern matching approaches to find valid JSON
                    # 1. Look for array pattern with objects
                    json_pattern1 = re.compile(r'(\[[\s]*\{.*?\}[\s]*\])', re.DOTALL)
                    # 2. Look for object pattern
                    json_pattern2 = re.compile(r'(\{[\s]*".*?"[\s]*:.*?\})', re.DOTALL)
                    # 3. Look for anything between square brackets
                    json_pattern3 = re.compile(r'(\[.*?\])', re.DOTALL)
                    
                    # Try patterns in order of specificity
                    found_valid_json = False
                    for pattern, pattern_name in [
                        (json_pattern1, "array with objects"), 
                        (json_pattern2, "object pattern"),
                        (json_pattern3, "square brackets")
                    ]:
                        match = pattern.search(content)
                        if match:
                            potential_json = match.group(1)
                            logger.info(f"Found potential JSON with {pattern_name} pattern: {potential_json[:100]}...")
                            # Try parsing the extracted content
                            try:
                                data = json.loads(potential_json)
                                logger.info(f"Successfully extracted JSON with {pattern_name} pattern")
                                found_valid_json = True
                                break  # Exit loop if successful
                            except json.JSONDecodeError as e2:
                                logger.warning(f"Failed to parse extracted JSON with {pattern_name} pattern: {str(e2)}")
                                continue  # Try next pattern
                    
                    if not found_valid_json:
                        # This executes if no patterns worked
                        logger.error("Could not find valid JSON patterns in the content")
                        raise ValueError("No valid JSON pattern found in content")
                
                if isinstance(data, list):
                    test_data = data
                else:
                    test_data = [data]  # Single item case
                    
                data_received = True
            except Exception as e:
                logger.error(f"Error processing JSON input: {str(e)}")
                return jsonify({'error': f'خطأ في معالجة مدخلات JSON: {str(e)}'}), 400
        
        # Handle direct JSON data from API
        elif request.is_json:
            try:
                data = request.json
                if isinstance(data, list):
                    test_data = data
                    data_received = True
                else:
                    return jsonify({'error': 'صيغة البيانات غير صحيحة. يجب أن تكون البيانات على شكل مصفوفة من الأسئلة والأجوبة.'}), 400
            except Exception as e:
                logger.error(f"Error processing JSON request: {str(e)}")
                return jsonify({'error': f'خطأ في معالجة بيانات JSON: {str(e)}'}), 400
        
        if not data_received:
            return jsonify({'error': 'لم يتم استلام أي بيانات للتقييم'}), 400
        
        # Standardize data format - handle both tuples (from CSV) and dictionaries (from JSON)
        # Expected formats: 
        # - [{"q": "question", "a": "answer"}, ...] (JSON format)
        # - [(question, answer), ...] (tuples format from CSV)
        questions = []
        expected_answers = []
        
        for item in test_data:
            if isinstance(item, tuple) and len(item) >= 2:
                # Handle tuple format (from CSV processing)
                questions.append(item[0])
                expected_answers.append(item[1])
            elif isinstance(item, dict):
                # Check for q/a format
                if 'q' in item and 'a' in item:
                    questions.append(item['q'])
                    expected_answers.append(item['a'])
                # Check for question/answer format
                elif 'question' in item and 'answer' in item:
                    questions.append(item['question'])
                    expected_answers.append(item['answer'])
                # Check for input/output format
                elif 'input' in item and 'output' in item:
                    questions.append(item['input'])
                    expected_answers.append(item['output'])
                # Check for prompt/expected format
                elif 'prompt' in item and 'expected' in item:
                    questions.append(item['prompt'])
                    expected_answers.append(item['expected'])
                # Unsupported format
                else:
                    logger.warning(f"Skipped item with unsupported format: {item}")
            else:
                logger.warning(f"Skipped item with unknown type: {type(item).__name__}")
        
        if not questions:
            return jsonify({
                'error': 'لم يتم العثور على أسئلة صالحة في البيانات المقدمة. ' +
                         'تأكد من استخدام التنسيق الصحيح: [{"q": "السؤال", "a": "الإجابة"}, ...]'
            }), 400
        
        logger.info(f"Prepared {len(questions)} questions for evaluation")
        
        # Define a generator function for streaming results
        def generate():
            try:
                # Initialize agent chain
                chunks = load_chunks()
                embedding_processor = EmbeddingProcessor()
                agent_chain = EnhancedAgentChain(chunks, embedding_processor)
                
                # Define callback for streaming results
                def stream_callback(result, metrics):
                    # Format as SSE event
                    data = {
                        "result": result,
                        "metrics": metrics
                    }
                    return f"data: {json.dumps(data)}\n\n"
                
                # Yield initial event
                yield f"data: {json.dumps({'status': 'started', 'total': len(questions)})}\n\n"
                
                # Fix: Use a collector list for results instead of nested yielding
                collected_results = []
                
                # Callback to collect results without yielding
                def collect_result(result, metrics):
                    collected_results.append((result, metrics))
                
                # Process the questions, which will collect results for us
                results, metrics = process_json_batch(
                    questions, 
                    expected_answers, 
                    agent_chain,
                    batch_size=3,  # Smaller batch size for more responsive updates
                    callback=collect_result
                )
                
                # Now stream the collected results
                for result, metrics in collected_results:
                    yield stream_callback(result, metrics)
                
                # Yield final event with complete results
                final_data = {
                    "status": "completed",
                    "results": results,
                    "metrics": metrics
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming evaluation: {str(e)}")
                logger.exception("Exception stacktrace:")
                import traceback
                error_data = {
                    "status": "error",
                    "error": f"حدث خطأ أثناء معالجة البيانات: {str(e)}",
                    "details": traceback.format_exc()
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        # Return streaming response
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'  # Disable buffering for Nginx
            }
        )
        
    except Exception as e:
        logger.error(f"Error setting up streaming evaluation: {str(e)}")
        return jsonify({'error': f'خطأ في إعداد التقييم: {str(e)}'}), 500

@app.route('/evaluate_simple', methods=['GET', 'POST'])
@login_required
def evaluate_simple():
    """Simple evaluation page that works reliably with Arabic text"""
    # Check if user is authenticated
    if not current_user.is_authenticated:
        flash('يرجى تسجيل الدخول للوصول إلى صفحة تقييم النظام', 'warning')
        return redirect(url_for('login', next=request.url))
    
    # Only allow superuser (test) to access evaluation page
    if not (current_user.username == 'test' or current_user.id == 1):
        flash('عذراً، صفحة تقييم النظام متاحة فقط للمشرفين', 'danger')
        return redirect(url_for('index'))
    
    logger.info(f"[Evaluate Simple] Access by admin user {current_user.username} (ID: {current_user.id})")
    
    # Results to pass to template
    results = None
    
    if request.method == 'POST':
        try:
            # Get questions and answers from form arrays
            questions = request.form.getlist('questions[]')
            expected_answers = request.form.getlist('answers[]')
            
            # Validate that we have matching questions and answers
            if len(questions) != len(expected_answers) or len(questions) == 0:
                flash('عدد الأسئلة والإجابات غير متطابق أو لا توجد أسئلة', 'danger')
                return redirect(url_for('evaluate_simple'))
            
            # Initialize agent chain
            agent_chain = EnhancedAgentChain(get_chunks())
            
            # Process questions
            predictions = []
            json_results = []
            
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
            
            for i, (question, expected) in enumerate(zip(questions, expected_answers)):
                logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
                
                # Try up to 3 times with increasing delays if we get an error
                max_retries = 3
                retry_count = 0
                answer = None
                
                while retry_count < max_retries:
                    try:
                        # Process the question with the agent chain
                        answer, used_chunks = agent_chain.process_question(question)
                        
                        # Check if the response contains error indicators
                        if is_error_response(answer):
                            logger.warning(f"Received error response for question {i+1}, retrying...")
                            retry_count += 1
                            # Exponential backoff - wait longer each retry
                            time.sleep(1 * (2 ** retry_count))
                            continue  # Try again
                        else:
                            break  # Success, exit retry loop
                    except Exception as e:
                        logger.error(f"Error processing question {i+1}: {str(e)}")
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.info(f"Retrying question {i+1} after error")
                            time.sleep(1 * (2 ** retry_count))
                        else:
                            answer = "عذراً، حدث خطأ أثناء معالجة هذا السؤال."
                            break
                
                # Add to predictions
                predictions.append(answer)
                
                # Create result object
                json_results.append({
                    'id': i + 1,
                    'question': question,
                    'ground_truth': expected,
                    'prediction': answer
                })
            
            # Import the metrics calculation function
            from evaluation import calculate_metrics_batch
            
            # Calculate metrics
            metrics = calculate_metrics_batch(predictions, expected_answers)
            
            # Add scores to results
            for i, (result, bleu, bert, llm) in enumerate(zip(
                json_results, 
                metrics['bleu_scores'], 
                metrics['bert_scores'], 
                metrics['llm_scores']
            )):
                result['bleu_score'] = bleu
                result['bert_score'] = bert
                result['llm_score'] = llm
            
            # Create final results object
            results = {
                'metrics': {
                    'total_questions': len(questions),
                    'average_bleu_score': metrics['average_bleu_score'],
                    'average_bert_score': metrics['average_bert_score'],
                    'average_llm_score': metrics['average_llm_score']
                },
                'results': json_results
            }
            
            logger.info(f"Completed evaluation with {len(questions)} questions")
            logger.info(f"Average scores - BLEU: {metrics['average_bleu_score']:.3f}, BERT: {metrics['average_bert_score']:.3f}, LLM: {metrics['average_llm_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            flash(f'خطأ أثناء التقييم: {str(e)}', 'danger')
            return redirect(url_for('evaluate_simple'))
    
    return render_template('evaluate_simple.html', results=results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)