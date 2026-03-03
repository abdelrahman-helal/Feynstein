# Firebase Setup Guide

## 🔧 Fix "Unauthorized Domain" Error

### **Option 1: Add Domain to Firebase Console (Recommended)**

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project: `sample-firebase-ai-app-846f5`
3. Navigate to **Authentication** → **Settings** → **Authorized domains**
4. Add these domains:
   - `localhost`
   - `127.0.0.1`
   - `localhost:5000` (if using Flask's default port)

### **Option 2: Use Firebase Auth Emulator (Local Development)**

For local development without domain restrictions:

1. **Install Firebase CLI:**
   ```bash
   npm install -g firebase-tools
   ```

2. **Login to Firebase:**
   ```bash
   firebase login
   ```

3. **Start Firebase Auth Emulator:**
   ```bash
   firebase emulators:start --only auth
   ```

4. **Set environment variable:**
   ```bash
   # Add to your .env file
   USE_FIREBASE_EMULATOR=true
   FLASK_ENV=development
   ```

### **Option 3: Quick Fix for Testing**

If you just want to test quickly, you can temporarily disable domain checking by adding this to your `.env` file:

```bash
# Add to .env file
FLASK_ENV=development
USE_FIREBASE_EMULATOR=true
```

Then run the Firebase emulator as shown in Option 2.

## 🚀 Production Deployment

When deploying to Vercel or other platforms, add your production domain to Firebase:

1. **Vercel:** Add `your-app-name.vercel.app`
2. **Custom Domain:** Add your custom domain
3. **Other Platforms:** Add your platform's domain

## 📝 Environment Variables

Make sure your `.env` file has all required variables:

```bash
# Firebase Configuration
FIREBASE_API_KEY=your_api_key
FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
FIREBASE_PROJECT_ID=your_project_id
FIREBASE_STORAGE_BUCKET=your_project.firebasestorage.app
FIREBASE_MESSAGING_SENDER_ID=your_sender_id
FIREBASE_APP_ID=your_app_id

# Development (optional)
FLASK_ENV=development
USE_FIREBASE_EMULATOR=true

# Other APIs
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=feynstein-db
```

## 🔍 Troubleshooting

- **"Unauthorized domain"**: Add domain to Firebase Console
- **"Project ID required"**: Set `GOOGLE_CLOUD_PROJECT` environment variable
- **Emulator not working**: Make sure Firebase CLI is installed and emulator is running 