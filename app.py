import os
from dotenv import load_dotenv
from flask import Flask
from flask_login import LoginManager
from user_manager import user_manager
from auth import auth
from main import main
from utils.langchain_rag_system import langchain_rag

# Load environment variables
load_dotenv()

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
    
    # Firebase configuration
    app.config['FIREBASE_API_KEY'] = os.getenv('FIREBASE_API_KEY')
    app.config['FIREBASE_AUTH_DOMAIN'] = os.getenv('FIREBASE_AUTH_DOMAIN')
    app.config['FIREBASE_PROJECT_ID'] = os.getenv('FIREBASE_PROJECT_ID')
    app.config['FIREBASE_STORAGE_BUCKET'] = os.getenv('FIREBASE_STORAGE_BUCKET')
    app.config['FIREBASE_MESSAGING_SENDER_ID'] = os.getenv('FIREBASE_MESSAGING_SENDER_ID')
    app.config['FIREBASE_APP_ID'] = os.getenv('FIREBASE_APP_ID')
    
    # Development environment flags
    app.config['FLASK_ENV'] = os.getenv('FLASK_ENV', 'production')
    # Convert string to boolean for template use
    use_emulator_str = os.getenv('USE_FIREBASE_EMULATOR', 'false').lower()
    app.config['USE_FIREBASE_EMULATOR'] = use_emulator_str == 'true'
    
    # Validate Firebase configuration
    if not app.config['FIREBASE_PROJECT_ID']:
        print("Warning: FIREBASE_PROJECT_ID not set. Firebase Auth may not work properly.")
        print("Please set FIREBASE_PROJECT_ID in your environment variables.")
    
    if not app.config['FIREBASE_API_KEY']:
        print("Warning: FIREBASE_API_KEY not set. Client-side Firebase may not work properly.")
        print("Please set FIREBASE_API_KEY in your environment variables.")
    
    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        """Load user by Firebase UID. Returns None if user doesn't exist or ID is invalid."""
        # Handle old database IDs (like "1") gracefully
        if not user_id or user_id.isdigit():
            # Old database ID or invalid ID - return None to force re-login
            return None
        
        # Try to get user by Firebase UID
        user = user_manager.get_user_by_firebase_uid(user_id)
        return user
    
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

# Vercel handler
app = create_app()

if __name__ == '__main__':
    app.run(debug=True) 