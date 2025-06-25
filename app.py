import os
from dotenv import load_dotenv
from flask import Flask
from flask_login import LoginManager
from models import db, User
from auth import auth
from main import main
from utils.initialize_rag import initialize_rag_system
from utils.rag_system import physics_rag

# Load environment variables
load_dotenv()

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///physics_tutor.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions
    db.init_app(app)
    
    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Register blueprints
    app.register_blueprint(auth, url_prefix='/auth')
    app.register_blueprint(main)
    
    # Initialize RAG system
    initialize_rag_system(physics_rag)
    
    return app

def init_db():
    """Initialize the database with tables"""
    with create_app().app_context():
        db.create_all()
        print("Database tables created successfully!")

if __name__ == '__main__':
    app = create_app()
    
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
    
    app.run(debug=True) 