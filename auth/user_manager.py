import os
from datetime import datetime
from typing import Optional, Dict, Any
from firebase_config import initialize_firebase, get_user_info
from firebase_admin import auth
import json

class FirebaseUser:
    """Firebase-based user class that mimics Flask-Login UserMixin"""
    
    def __init__(self, firebase_uid: str, email: str, display_name: str = None, photo_url: str = None):
        self.firebase_uid = firebase_uid
        self.email = email
        self.display_name = display_name or ""
        self.photo_url = photo_url or ""
        self.is_authenticated = True
        self.is_active = True
        self.is_anonymous = False
    
    def get_id(self):
        """Return the firebase_uid as the user ID"""
        return self.firebase_uid
    
    def __repr__(self):
        return f'<FirebaseUser {self.email}>'

class UserManager:
    """Manages users using Firebase instead of SQLAlchemy"""
    
    def __init__(self):
        initialize_firebase()
    
    def get_user_by_firebase_uid(self, firebase_uid: str) -> Optional[FirebaseUser]:
        """Get user by Firebase UID. Returns None if user doesn't exist."""
        if not firebase_uid:
            return None
            
        try:
            user_info = get_user_info(firebase_uid)
            if user_info:
                return FirebaseUser(
                    firebase_uid=user_info['uid'],
                    email=user_info['email'],
                    display_name=user_info.get('display_name'),
                    photo_url=user_info.get('photo_url')
                )
        except Exception as e:
            # User doesn't exist in Firebase - this is okay, they'll need to sign in again
            # Don't print error for "user not found" as it's expected for invalid sessions
            if 'No user record found' not in str(e):
                print(f"Error getting user by Firebase UID: {e}")
        return None
    
    def create_or_update_user(self, firebase_uid: str, email: str, display_name: str = None, photo_url: str = None) -> FirebaseUser:
        """Create or update user in Firebase (user data is managed by Firebase)
        
        Firebase OAuth automatically creates users on first sign-in, so this method:
        - For existing users: Tries to get latest info from Firebase Admin SDK
        - For new users: Uses token data (which is always available)
        - Always returns a valid FirebaseUser object
        """
        # Try to get user info from Firebase Admin SDK (for existing users)
        user_info = None
        try:
            user_info = get_user_info(firebase_uid)
        except Exception as e:
            # User might not exist in Admin SDK yet (new user) or Admin SDK unavailable
            # This is fine - we'll use token data instead
            pass
        
        # Use Firebase Admin SDK data if available, otherwise use token data
        if user_info:
            # Existing user - use data from Firebase Admin SDK
            return FirebaseUser(
                firebase_uid=user_info['uid'],
                email=user_info.get('email') or email,
                display_name=user_info.get('display_name') or display_name,
                photo_url=user_info.get('photo_url') or photo_url
            )
        else:
            # New user or Admin SDK unavailable - use token data
            # Firebase OAuth has already created the user, we just need to create our user object
            return FirebaseUser(
                firebase_uid=firebase_uid,
                email=email,
                display_name=display_name or "",
                photo_url=photo_url or ""
            )

# Global user manager instance
user_manager = UserManager() 