from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_login import current_user, login_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from enhanced_langchain_agents import initialize_enhanced_system
from app import db, login_manager
from models import User, MinistryContact, ChatHistory
from config import Config
from dotenv import load_dotenv
import logging
import os
import secrets
import json
from datetime import datetime
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    CORS(app)
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    return app

# Create the app
app = create_app()

@login_manager.user_loader
def load_user(user_id):
    from models import User
    return User.query.get(int(user_id))

# Initialize the enhanced system
enhanced_system = initialize_enhanced_system()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/chat', methods=['GET'])
def chat():
    return render_template('index.html')

@app.route('/contacts')
def contacts():
    """Show ministry contacts"""
    # Get all contacts from database
    contacts = MinistryContact.query.all()
    
    # If no contacts exist, create some default ones
    if not contacts:
        contacts = [
            MinistryContact(
                name="وزارة العمل الأردنية - المكتب الرئيسي",
                position="المكتب الرئيسي",
                phone="065530000",
                email="info@mol.gov.jo",
                address="عمان، شارع الملكة نور"
            ),
            MinistryContact(
                name="وزارة العمل - مكتب عمان",
                position="مكتب عمان",
                phone="064643336",
                email="amman@mol.gov.jo",
                address="عمان، وسط البلد"
            ),
            MinistryContact(
                name="وزارة العمل - مكتب إربد",
                position="مكتب إربد",
                phone="027243856",
                email="irbid@mol.gov.jo",
                address="إربد، شارع فلسطين"
            )
        ]
        db.session.add_all(contacts)
        db.session.commit()
        contacts = MinistryContact.query.all()
    
    return render_template('contacts.html', contacts=contacts)

@app.route('/history')
@login_required
def history():
    return render_template('history.html')

@app.route('/history/<int:history_id>')
@login_required
def history_detail(history_id):
    return render_template('history_detail.html', history_id=history_id)

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember_me = 'remember_me' in request.form
        
        if not username or not password:
            flash('Please enter username and password', 'danger')
            return redirect(url_for('login'))
        
        from models import User
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=remember_me)
            next_page = request.args.get('next')
            if not next_page or not next_page.startswith('/'):
                next_page = url_for('index')
            return redirect(next_page)
        else:
            flash('Invalid username or password', 'danger')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not password:
            flash('Please fill all fields', 'danger')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        
        from models import User
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        new_user = User(username=username)
        new_user.set_password(password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! You can now login', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating user: {str(e)}")
            flash('Error creating account. Please try again', 'danger')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))

@app.route('/api/query', methods=['POST'])
@login_required
def process_query():
    if enhanced_system is None:
        return jsonify({
            'error': 'System not initialized',
            'message': 'The system is not ready yet'
        }), 503

    try:
        # Try to get data from JSON first
        data = request.get_json(silent=True)
        
        # If no JSON data, try form data
        if not data:
            query = request.form.get('query')
        else:
            query = data.get('query', '')
            
        if not query:
            return jsonify({
                'error': 'No query provided',
                'message': 'Please provide a query in the request body or form data'
            }), 400
        
        # Process the query using the enhanced system
        response = enhanced_system.process_query(query)
        
        # Save to chat history if user is authenticated
        if current_user.is_authenticated:
            history = ChatHistory(
                user_id=current_user.id,
                query=query,
                response=response['answer'],
                sources=json.dumps(response.get('sources', [])),
                metrics=json.dumps(response.get('metrics', {}))
            )
            db.session.add(history)
            db.session.commit()
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/ask', methods=['POST'])
def ask():
    """Compatibility endpoint for existing frontend that uses the /ask route"""
    try:
        # Extract question from request
        data = request.json
        question = data.get('question', '')
        conversation_id = data.get('conversation_id')
        
        if not question:
            return jsonify({'error': 'الرجاء إدخال سؤال'}), 400
            
        logger.info(f"Processing question via /ask endpoint: '{question}'")
        
        # Process the question using the enhanced system
        response = enhanced_system.process_query(question)
        answer = response['answer']
        sources = response.get('sources', [])
        
        # Format sources to match the expected format in the frontend
        chunks_for_response = []
        for source in sources:
            chunk_obj = {
                'text': source.get('text', ''),
                'article': source.get('article', ''),
                'source': 'قانون العمل الأردني',
                'metadata': {
                    'article': source.get('article', ''),
                    'source': 'قانون العمل الأردني',
                    'display_title': source.get('article', 'مصدر قانوني')
                }
            }
            chunks_for_response.append(chunk_obj)
        
        # Save to conversation if conversation_id is provided
        if conversation_id and current_user.is_authenticated:
            from models import Conversation, Message
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
        
        return jsonify({
            'answer': answer,
            'chunks': chunks_for_response,
            'chunk_count': len(chunks_for_response),
            'debug': {
                'has_chunks': len(chunks_for_response) > 0,
                'sample_chunk': chunks_for_response[0] if chunks_for_response else None
            }
        })
    except Exception as e:
        logger.error(f"Error in /ask endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'answer': "عذراً، حدث خطأ أثناء معالجة السؤال. يرجى المحاولة مرة أخرى لاحقاً.",
            'chunks': [],
            'chunk_count': 0,
            'debug': {'error': True, 'reason': f'Exception: {str(e)}', 'has_chunks': False}
        })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/direct_json_test', methods=['GET'])
@login_required
def direct_json_test():
    """Test page for direct JSON evaluation"""
    return render_template('arabic_json_form.html')

@app.route('/test_arabic_json', methods=['GET'])
@login_required
def test_arabic_json():
    """Test page for Arabic JSON evaluation"""
    return render_template('arabic_json_form.html')

@app.route('/api/chat/history', methods=['GET'])
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

@app.route('/api/chat/clear', methods=['POST'])
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

if __name__ == '__main__':
    # Ensure the JSON file exists
    if not os.path.exists(LABOR_LAW_JSON):
        logger.error(f"Labor law JSON file not found: {LABOR_LAW_JSON}")
        raise FileNotFoundError(f"Required file not found: {LABOR_LAW_JSON}")
    
    app.run(debug=True) 