import logging
import os
import secrets
from sqlalchemy.orm import DeclarativeBase
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

# logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass

# Initialize 
db = SQLAlchemy(model_class=Base)
login_manager = LoginManager()
login_manager.login_view = 'login'

# Create the Flask application
def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ.get("SESSION_SECRET", secrets.token_hex(16))
    
    # configure database
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }
    
    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    
    # initialize tables within app context
    with app.app_context():
        try:
            # import models here to avoid circular imports issue 
            import models  
            db.create_all()
            logger.info("Database tables initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    return app