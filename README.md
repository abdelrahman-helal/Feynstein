# Feynstein - AI Physics Tutor

An intelligent AI-powered physics tutoring application that combines the power of Retrieval-Augmented Generation (RAG) with advanced language models to provide contextually accurate physics education. Built with LangChain, Google Gemini, and Qwen vision models for comprehensive learning support.

## Features

### 🧠 **Intelligent RAG System**
- **Retrieval-Augmented Generation**: Combines textbook knowledge with AI reasoning
- **Physics Textbook Integration**: Pre-processed physics textbooks provide accurate context
- **Context-Aware Responses**: AI responses are grounded in verified physics content
- **Smart Context Selection**: Automatically retrieves relevant textbook sections for questions

### 🤖 **AI Models**
- **Google Gemini 2.0 Flash**: Primary physics tutoring with high accuracy and reasoning
- **Qwen2.5-VL Vision Model**: Advanced image processing for equation recognition
- **LangChain Integration**: Orchestrates multiple AI models for optimal responses
- **Conversation Memory**: Maintains context across chat sessions

### 💬 **Interactive Learning**
- **Text Questions**: Ask physics questions in natural language
- **Image Questions**: Upload equation images for instant analysis
- **Step-by-step Guidance**: Interactive problem-solving with detailed explanations
- **Follow-up Questions**: Seamless conversation flow without losing context

### 🔐 **User Experience**
- **Secure Authentication**: User registration and login system
- **Chat History**: Persistent conversations with searchable history
- **Modern UI**: ChatGPT-like interface with dark mode support
- **Math Rendering**: Beautiful LaTeX equation display with MathJax
- **Responsive Design**: Works seamlessly on desktop and mobile devices

**Required API Keys:**
- **Google API Key**: For Gemini 2.0 Flash model access
- **Hugging Face Token**: For image processing with Qwen2.5-VL model
- **Pinecone API Key**: For vector database storage (optional, can use local Chroma)
- **Secret Key**: For Flask session management

### Initialize Database

```bash
python init_db.py
```

This will create the PostgreSQL database tables and a default admin user:
- Username: `admin`
- Password: `admin123`

**Note**: Make sure your PostgreSQL database is running and accessible via the DATABASE_URL environment variable.

### Process Textbooks (One-time setup)

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

## Project Structure

```
Feynstein/
├── app.py                     # Main Flask application with PostgreSQL support
├── auth.py                   # Authentication blueprint
├── main.py                   # Main chat functionality with RAG integration
├── models.py                 # Database models (User, Chat, Message)
├── init_db.py                # Database initialization script
├── process_textbooks.py      # Textbook processing and vectorization
├── requirements.txt          # Python dependencies including PostgreSQL
├── Procfile                  # Production deployment configuration
├── env.example               # Environment variables template
├── chroma_db/                # Chroma vector database (generated)
├── textbooks/                # Physics textbook PDFs for RAG
├── templates/
│   ├── auth/
│   │   ├── login.html        # Login page
│   │   └── signup.html       # Signup page
│   └── chat.html             # Main chat interface with MathJax
└── utils/
    ├── langchain_model_handler.py    # LangChain integration with Gemini 2.0 Flash
    ├── langchain_rag_system.py       # RAG system with Chroma vector database
    └── equation_processor.py         # Image processing utilities
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

**Note**: The application now uses PostgreSQL for production deployment, providing better scalability and performance compared to SQLite.

## Technologies Used

- **Backend**: Flask, SQLAlchemy, Flask-Login, Gunicorn
- **AI/ML**: LangChain, Google Gemini 2.0 Flash, Qwen2.5-VL, sentence-transformers
- **Frontend**: HTML, CSS (Tailwind), JavaScript, MathJax
- **Database**: PostgreSQL (production)
- **Vector Database**: Pinecone
- **Message Persistence**: LangGraph checkpoints with conversation memory
- **Production**: Render-ready with Procfile and environment configuration




