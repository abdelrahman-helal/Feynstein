import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, auth
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# Load environment variables
load_dotenv()

# Firebase configuration from environment variables
FIREBASE_CONFIG = {
    "apiKey": os.getenv('FIREBASE_API_KEY'),
    "authDomain": os.getenv('FIREBASE_AUTH_DOMAIN'),
    "projectId": os.getenv('FIREBASE_PROJECT_ID'),
    "storageBucket": os.getenv('FIREBASE_STORAGE_BUCKET'),
    "messagingSenderId": os.getenv('FIREBASE_MESSAGING_SENDER_ID'),
    "appId": os.getenv('FIREBASE_APP_ID')
}

# Initialize Firebase Admin SDK
def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    try:
        # Check if Firebase is already initialized
        firebase_admin.get_app()
    except ValueError:
        # Get project ID from environment
        project_id = os.getenv('FIREBASE_PROJECT_ID')
        if not project_id:
            print("Warning: FIREBASE_PROJECT_ID not set. Using default project.")
            project_id = None
        
        # Use Firebase Auth Emulator for local development
        use_emulator = os.getenv('FLASK_ENV') == 'development' or os.getenv('USE_FIREBASE_EMULATOR', 'false').lower() == 'true'
        
        if use_emulator:
            # Set environment variable for emulator (this tells Admin SDK to use emulator)
            os.environ['FIREBASE_AUTH_EMULATOR_HOST'] = 'localhost:9099'
            # Initialize with minimal config - emulator doesn't validate credentials
            # We'll suppress credential errors since emulator works without them
            try:
                firebase_admin.initialize_app(options={
                    'projectId': project_id or 'demo-project'
                })
            except Exception as init_error:
                # If initialization fails, try with a credential object
                # This is a workaround for Admin SDK requiring credentials
                print(f"Note: Using emulator mode (credential validation disabled)")
                # Re-raise only if it's not a credential-related error
                if 'credential' not in str(init_error).lower():
                    raise
                # For emulator, we can continue without proper credentials
                # The emulator will handle authentication
                pass
            print("Firebase Auth Emulator enabled for local development")
        else:
            # Production: Use real credentials
            if os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY'):
                # Use service account key from environment variable (path to JSON file)
                service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY')
                if os.path.exists(service_account_path):
                    cred = credentials.Certificate(service_account_path)
                else:
                    print(f"Warning: Service account file not found at {service_account_path}")
                    print("Falling back to Application Default Credentials")
                    cred = credentials.ApplicationDefault()
            else:
                # Use default credentials (for production)
                # This works if you've set up gcloud auth or are running on GCP
                try:
                    cred = credentials.ApplicationDefault()
                except Exception as e:
                    print(f"Warning: Could not load Application Default Credentials: {e}")
                    print("For local production testing, you may need to:")
                    print("1. Download a service account key from Firebase Console")
                    print("2. Set FIREBASE_SERVICE_ACCOUNT_KEY environment variable to the key file path")
                    print("3. Or use the Firebase Auth Emulator for local development")
                    raise
            
            firebase_admin.initialize_app(cred, {
                'projectId': project_id
            })
            print("Firebase initialized in production mode")

def verify_firebase_token(id_token_string):
    """Verify Firebase ID token"""
    try:
        # Verify the token
        decoded_token = auth.verify_id_token(id_token_string)
        return decoded_token
    except Exception as e:
        print(f"Token verification failed: {e}")
        return None

def get_user_info(uid):
    """Get user information from Firebase Admin SDK
    
    Returns user info if user exists, None otherwise.
    This is used to get additional user data, but token data is always available as fallback.
    """
    if not uid:
        return None
        
    try:
        user = auth.get_user(uid)
        return {
            'uid': user.uid,
            'email': user.email,
            'display_name': user.display_name,
            'photo_url': user.photo_url
        }
    except Exception as e:
        # Handle different error cases
        error_str = str(e)
        
        # User doesn't exist - this is normal for new users or invalid sessions
        if 'No user record found' in error_str:
            return None
        
        # When using emulator, get_user might fail due to credential issues
        # This is okay - we'll use the token data instead
        use_emulator = os.getenv('FLASK_ENV') == 'development' or os.getenv('USE_FIREBASE_EMULATOR', 'false').lower() == 'true'
        if use_emulator:
            # In emulator mode, we can't always get user info via Admin SDK
            # This is expected - we'll rely on token data from client
            return None
        else:
            # Only log unexpected errors
            if 'credential' not in error_str.lower():
                print(f"Failed to get user info: {e}")
            return None 