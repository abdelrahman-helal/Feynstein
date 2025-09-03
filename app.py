import os
from dotenv import load_dotenv
from flask import Flask
from flask_login import LoginManager
from models import db, User
from auth import auth
from main import main
from utils.langchain_rag_system import langchain_rag

# Load environment variables
load_dotenv()

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
    
    # Use environment variable for database URI in production, fallback to hardcoded for development
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        # Render provides DATABASE_URL, but SQLAlchemy expects DATABASE_URL
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        app.config['SQLALCHEMY_DATABASE_URI'] = database_url    
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
        return db.session.get(User, int(user_id))
    
    # Register blueprints
    app.register_blueprint(auth, url_prefix='/auth')
    app.register_blueprint(main)
    
    # Initialize LangChain RAG system (only in development)
    if not os.getenv('VERCEL'):
        with app.app_context():
            # Try to load existing vector database
            if not langchain_rag.load_vector_db():
                print("No existing vector database found. Please run 'python process_textbooks.py' to create one.")
                print("The app will work without RAG functionality until the vector database is created.")
    
    return app

def init_db():
    """Initialize the database with tables"""
    with create_app().app_context():
        db.create_all()
        print("Database tables created successfully!")

# Vercel handler
app = create_app()

# Create database tables if they don't exist
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True) 