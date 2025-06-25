# Feynstein

An AI-powered physics tutoring application with authentication, chat history, and support for both text and image-based questions. Built with LangChain, Gemini 2.0 Flash, and modern web technologies.

## Features

- **User Authentication**: Secure sign-up and login system
- **Chat History**: Persistent chat sessions with message persistence using LangGraph checkpoints
- **Text Questions**: Ask physics questions in text format
- **Image Questions**: Upload images of equations for analysis
- **Step-by-step Guidance**: Interactive learning with step-by-step solutions
- **RAG System**: Retrieval-Augmented Generation using pre-processed physics textbooks
- **Modern UI**: ChatGPT-like interface with responsive design
- **LangChain Integration**: Built with LangChain and Gemini 2.0 Flash for enhanced AI capabilities

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
HF_TOKEN=your_huggingface_token_here
GOOGLE_API_KEY=your_google_api_key_here
SECRET_KEY=your_secret_key_here
```

**Required API Keys:**
- **Google API Key**: For Gemini 2.0 Flash model access
- **Hugging Face Token**: For image processing with Qwen model
- **Secret Key**: For Flask session management

### 3. Initialize Database

```bash
python init_db.py
```

This will create the SQLite database and a default admin user:
- Username: `admin`
- Password: `admin123`

### 4. Process Textbooks (One-time setup)

Add your physics textbook PDFs to the `textbooks/` directory, then run:

```bash
python process_textbooks.py
```

This will:
- Process all PDF textbooks in the `textbooks/` directory
- Create embeddings using sentence-transformers
- Save the vector database as a Chroma database in `chroma_db/` directory
- Enable RAG functionality for physics questions

**Note**: This step is required for RAG functionality. The app will work without it, but won't have access to textbook knowledge.

### 5. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

### Authentication
1. Visit the application
2. Sign up for a new account or log in with existing credentials
3. You'll be redirected to the main chat interface

### Using the Chat Interface
1. **New Chat**: Click the "+" button in the sidebar to start a new conversation
2. **Text Questions**: Type your physics question in the text area
3. **Image Questions**: Click the "Image" button and upload an equation image
4. **Chat History**: Click on any chat in the sidebar to view previous conversations
5. **Delete Chats**: Use the trash icon to delete unwanted conversations

### Features
- **Real-time Responses**: Get immediate AI-powered physics explanations using Gemini 2.0 Flash
- **Step-by-step Learning**: Interactive guidance through problem-solving
- **Context-Aware**: Uses relevant textbook information for accurate responses
- **Math Rendering**: Supports LaTeX math equations with MathJax
- **Message Persistence**: Each chat maintains conversation history using LangGraph checkpoints

## Project Structure

```
Feynstein/
├── app.py                     # Main Flask application
├── auth.py                   # Authentication blueprint
├── main.py                   # Main chat functionality blueprint
├── models.py                 # Database models
├── init_db.py                # Database initialization script
├── process_textbooks.py      # Textbook processing script
├── requirements.txt          # Python dependencies
├── chroma_db/                # Chroma vector database (generated)
├── templates/
│   ├── auth/
│   │   ├── login.html        # Login page
│   │   └── signup.html       # Signup page
│   └── chat.html             # Main chat interface
├── textbooks/                # Physics textbook PDFs
└── utils/
    ├── langchain_model_handler.py    # LangChain model with Gemini 2.0 Flash
    ├── langchain_rag_system.py       # RAG system with Chroma vector database
    ├── equation_processor.py         # Image processing utilities
    ├── initialize_rag.py             # Legacy RAG system (deprecated)
    ├── model_handler.py              # Legacy model handler (deprecated)
    └── rag_system.py                 # Legacy RAG system (deprecated)
```

## Database Schema

### Users
- `id`: Primary key
- `username`: Unique username
- `email`: Unique email address
- `password_hash`: Hashed password
- `created_at`: Account creation timestamp

### Chats
- `id`: Primary key
- `title`: Chat title (first question)
- `user_id`: Foreign key to User
- `created_at`: Chat creation timestamp
- `updated_at`: Last message timestamp

### Messages
- `id`: Primary key
- `chat_id`: Foreign key to Chat
- `role`: 'user' or 'assistant'
- `content`: Message content
- `timestamp`: Message timestamp
- `question_type`: 'text' or 'image'
- `context_used`: Whether RAG context was used

## Technologies Used

- **Backend**: Flask, SQLAlchemy, Flask-Login
- **AI/ML**: LangChain, Gemini 2.0 Flash, Qwen2.5-VL, sentence-transformers
- **Frontend**: HTML, CSS (Tailwind), JavaScript
- **Database**: SQLite3
- **Vector Database**: Chroma with persistent storage
- **Math Rendering**: MathJax
- **Authentication**: Werkzeug password hashing
- **Message Persistence**: LangGraph checkpoints

## Deployment to Vercel

### Prerequisites
1. Process textbooks locally using `python process_textbooks.py`
2. Ensure `chroma_db/` directory is included in your repository
3. Set up environment variables in Vercel dashboard

### Deployment Steps
1. Push your code to GitHub
2. Connect your repository to Vercel
3. Set environment variables:
   - `GOOGLE_API_KEY`
   - `HF_TOKEN`
   - `SECRET_KEY`
4. Deploy

**Important**: The `chroma_db/` directory contains the pre-processed textbook embeddings and should be included in your deployment.

## Security Features

- Password hashing with Werkzeug
- Session management with Flask-Login
- CSRF protection
- Input validation and sanitization
- Secure password requirements
- Thread-based message persistence

## API Integration

### Google Gemini 2.0 Flash
- Used for main physics tutoring responses
- Configured with temperature 0.7 for balanced creativity and accuracy
- Supports up to 2048 tokens per response

### Hugging Face Qwen2.5-VL
- Used for image-based equation extraction
- Processes uploaded equation images
- Extracts text representations for further processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License. 