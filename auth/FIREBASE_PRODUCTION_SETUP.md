# Firebase Production Mode Setup

This guide explains how to use Firebase in production mode (without the emulator).

## Quick Setup

### 1. Update Your `.env` File

Make sure your `.env` file has these settings:

```env
# Set to production (or remove FLASK_ENV entirely)
FLASK_ENV=production

# Disable emulator
USE_FIREBASE_EMULATOR=false

# Your Firebase configuration (already set)
FIREBASE_API_KEY=your_api_key
FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
FIREBASE_PROJECT_ID=your_project_id
FIREBASE_STORAGE_BUCKET=your_project.firebasestorage.app
FIREBASE_MESSAGING_SENDER_ID=your_sender_id
FIREBASE_APP_ID=your_app_id
```

### 2. Add Authorized Domains in Firebase Console

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project
3. Navigate to **Authentication** → **Settings** → **Authorized domains**
4. Add these domains:
   - `localhost` (for local testing)
   - `127.0.0.1` (for local testing)
   - Your production domain (when deploying)

### 3. Set Up Firebase Admin SDK Credentials

You have two options:

#### Option A: Service Account Key (Recommended for Local Development)

1. Go to Firebase Console → Project Settings → Service Accounts
2. Click "Generate New Private Key"
3. Download the JSON file
4. Save it in your project directory (e.g., `serviceAccountKey.json`)
5. Add to your `.env` file:
   ```env
   FIREBASE_SERVICE_ACCOUNT_KEY=./serviceAccountKey.json
   ```
   Or use the full path:
   ```env
   FIREBASE_SERVICE_ACCOUNT_KEY=D:\CS\Feynstein\serviceAccountKey.json
   ```

#### Option B: Application Default Credentials (For GCP/Production)

If you're deploying to Google Cloud Platform, you can use Application Default Credentials. Otherwise, use Option A.

### 4. Enable Google Sign-In Provider

1. Go to Firebase Console → Authentication → Sign-in method
2. Enable **Google** as a sign-in provider
3. Add your project's support email
4. Save

### 5. Run Your App

```bash
python app.py
```

The app will now use production Firebase (no emulator needed).

## Testing Locally with Production Firebase

You can test with production Firebase locally:

1. Make sure `USE_FIREBASE_EMULATOR=false` in your `.env`
2. Make sure `localhost` is in Firebase authorized domains
3. Download and configure the service account key
4. Run your Flask app

## Troubleshooting

### "Unauthorized Domain" Error

- Make sure `localhost` is added to Firebase authorized domains
- Check that your `FIREBASE_AUTH_DOMAIN` matches your Firebase project

### "Default credentials not found" Error

- Download a service account key from Firebase Console
- Set `FIREBASE_SERVICE_ACCOUNT_KEY` in your `.env` file
- Make sure the path to the key file is correct

### "Invalid token" Error

- Make sure your Firebase configuration in `.env` is correct
- Verify that Google Sign-In is enabled in Firebase Console

## Production Deployment

When deploying to production:

1. Add your production domain to Firebase authorized domains
2. Set environment variables in your hosting platform (Vercel, etc.)
3. Upload your service account key securely (or use environment variables)
4. Make sure `USE_FIREBASE_EMULATOR=false`

## Security Notes

- **Never commit** your `serviceAccountKey.json` file to git
- Add it to `.gitignore`
- Use environment variables for sensitive data
- Rotate service account keys periodically

