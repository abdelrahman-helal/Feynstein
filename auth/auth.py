from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import login_user, logout_user, login_required, current_user
from user_manager import user_manager
from firebase_config import verify_firebase_token
import os

auth = Blueprint('auth', __name__)

@auth.route('/login')
def login():
    """Show login page with Google Sign-In"""
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    return render_template('auth/login.html')

@auth.route('/google/callback', methods=['POST'])
def google_auth_callback():
    """Handle Google OAuth callback with Firebase token (for both popup and redirect flows)"""
    try:
        # Get the ID token from the request
        data = request.get_json()
        id_token_string = data.get('idToken')
        
        if not id_token_string:
            return jsonify({'error': 'No ID token provided'}), 400
        
        # Verify the Firebase token
        decoded_token = verify_firebase_token(id_token_string)
        if not decoded_token:
            return jsonify({'error': 'Invalid token'}), 401
        
        # Get user info from Firebase
        firebase_uid = decoded_token['uid']
        email = decoded_token.get('email', '')
        display_name = decoded_token.get('name', '')
        photo_url = decoded_token.get('picture', '')
        
        # Create or get user using Firebase
        user = user_manager.create_or_update_user(
            firebase_uid=firebase_uid,
            email=email,
            display_name=display_name,
            photo_url=photo_url
        )
        
        # Log in the user
        login_user(user)
        
        return jsonify({
            'success': True,
            'redirect': url_for('main.home')
        })
        
    except Exception as e:
        print(f"Authentication error: {e}")
        return jsonify({'error': 'Authentication failed'}), 500

@auth.route('/google/redirect-callback', methods=['GET'])
def google_redirect_callback():
    """Handle Google OAuth redirect callback (redirects back to login page which handles the result client-side)"""
    return redirect(url_for('auth.login'))

@auth.route('/logout')
@login_required
def logout():
    """Logout user"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))

@auth.route('/verify', methods=['POST'])
def verify_auth():
    """Verify if user is authenticated"""
    if current_user.is_authenticated:
        return jsonify({
            'authenticated': True,
            'user': {
                'email': current_user.email,
                'display_name': current_user.display_name,
                'photo_url': current_user.photo_url
            }
        })
    else:
        return jsonify({'authenticated': False}), 401 