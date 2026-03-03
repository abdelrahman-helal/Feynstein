from flask import Blueprint, render_template, request, jsonify, Response, stream_with_context
from flask_login import login_required, current_user
from utils.chat_storage import chat_storage
from utils.langchain_rag_system import langchain_rag
from utils.langchain_model_handler import feynstein_model
from utils.peer_observation_handler import peer_observation_handler
import base64
from huggingface_hub import InferenceClient
import os
import uuid
import json

main = Blueprint('main', __name__)

# Initialize Qwen client for image processing
qwen_client = InferenceClient(
    provider="hyperbolic",
    api_key=os.getenv("HF_TOKEN")
)

@main.route('/')
@login_required
def home():
    # Get user's chat history
    chats = chat_storage.get_user_chats(current_user.firebase_uid)
    # Sort by updated_at (newest first)
    chats.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
    return render_template('chat.html', chats=chats)

@main.route('/chats')
@login_required
def get_chats():
    chats = chat_storage.get_user_chats(current_user.firebase_uid)
    # Sort by updated_at (newest first)
    chats.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
    return jsonify([{
        'id': chat['id'],
        'title': chat['title'],
        'created_at': chat['created_at'],
        'updated_at': chat['updated_at'],
        'message_count': len(chat['messages'])
    } for chat in chats])

@main.route('/chats/<chat_id>')
@login_required
def get_chat(chat_id):
    chat = chat_storage.get_chat(current_user.firebase_uid, chat_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
    return jsonify({
        'id': chat['id'],
        'title': chat['title'],
        'messages': [{
            'id': msg['id'],
            'role': msg['role'],
            'content': msg['content'],
            'timestamp': msg['timestamp'],
            'question_type': msg['question_type']
        } for msg in chat['messages']]
    })

@main.route('/chats/new', methods=['POST'])
@login_required
def create_chat():
    data = request.json
    title = data.get('title', 'New Chat')
    
    chat = chat_storage.create_chat(current_user.firebase_uid, title)
    
    return jsonify({
        'id': chat['id'],
        'title': chat['title'],
        'created_at': chat['created_at']
    })

@main.route('/chats/<chat_id>/delete', methods=['DELETE'])
@login_required
def delete_chat(chat_id):
    success = chat_storage.delete_chat(current_user.firebase_uid, chat_id)
    if not success:
        return jsonify({'error': 'Chat not found'}), 404
    return jsonify({'message': 'Chat deleted successfully'})

@main.route('/process_question', methods=['POST'])
@login_required
def process_question():
    """
    Handle the first question in a new chat or conversation.
    This endpoint provides RAG context and creates new chat sessions.
    For follow-up questions, use the next_step endpoint instead.
    """
    data = request.json
    question_type = data.get('type')  # 'text' or 'image'
    content = data.get('content')
    chat_id = data.get('chat_id')
    
    # Create new chat if not provided
    if not chat_id:
        chat = chat_storage.create_chat(
            current_user.firebase_uid, 
            content[:50] + "..." if len(content) > 50 else content
        )
        chat_id = chat['id']
    else:
        chat = chat_storage.get_chat(current_user.firebase_uid, chat_id)
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404
    
    # Generate thread ID for this chat (using chat_id for consistency)
    thread_id = f"user_{current_user.firebase_uid}_chat_{chat_id}"
    
    # Save user message
    user_message = chat_storage.add_message(
        current_user.firebase_uid,
        chat_id,
        'user',
        content,
        question_type
    )
    
    if question_type == 'text':
        # Process text-based question with RAG
        context = langchain_rag.query_relevant_content(query=content)
        print(f'Context: {context}')
        response = feynstein_model.generate_response(
            question=content,
            thread_id=thread_id,
            context=context,
            question_type=question_type
        )
        
    elif question_type == 'image':
        try:
            # Process image-based question
            image_base64 = content
            # Remove the data URL prefix if present
            if image_base64.startswith('data:image'):
                image_base64 = image_base64.split(',', 1)[1]
            
            # First, get image description using Qwen
            qwen_response = qwen_client.chat.completions.create(
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Write the equation in this image in text form."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ]
            )
            extracted_equation = qwen_response.choices[0].message.content
            
            # Combine the image description with the equation
            combined_query = f"Extracted equation: {extracted_equation}"
            
            # Query relevant content using the combined information
            context = langchain_rag.query_relevant_content(combined_query)

            # Generate response with both the image description and the extracted equation
            response = feynstein_model.generate_response(
                question=f"Please help me understand this physics problem: {combined_query}",
                thread_id=thread_id,
                context=context,
                question_type=question_type
            )
            
        except Exception as e:
            response = {
                "explanation": f"Error processing image: {str(e)}",
                "next_steps": [],
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
    
    # Save assistant response
    assistant_message = chat_storage.add_message(
        current_user.firebase_uid,
        chat_id,
        'assistant',
        response['explanation'],
        'text',
        response.get('metadata', {}).get('context_used', False)
    )
    
    return jsonify({
        'chat_id': chat_id,
        'response': response,
        'user_message_id': user_message['id'],
        'assistant_message_id': assistant_message['id'],
        'thread_id': thread_id
    })

@main.route('/next_step', methods=['POST'])
@login_required
def next_step():
    """
    Handle follow-up questions in an existing chat.
    This endpoint is used for subsequent messages after the initial question.
    No additional RAG context is provided - the model relies on conversation history.
    """
    data = request.json
    current_step = data.get('current_step', 0)
    student_response = data.get('response')
    chat_id = data.get('chat_id')
    
    if not chat_id:
        return jsonify({'error': 'Chat ID required'}), 400
    
    chat = chat_storage.get_chat(current_user.firebase_uid, chat_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
    
    # Generate thread ID for this chat
    thread_id = f"user_{current_user.firebase_uid}_chat_{chat_id}"
    
    # Save user response
    user_message = chat_storage.add_message(
        current_user.firebase_uid,
        chat_id,
        'user',
        student_response
    )
    
    # Generate next step based on student's response using LangChain model
    # For follow-up questions, we don't need additional context from RAG
    # The model can use conversation history to provide contextual responses
    response = feynstein_model.generate_response(
        question=student_response,
        thread_id=thread_id,
        context="",  # No additional context for follow-up responses
        question_type="text"
    )
    
    # Save assistant response
    assistant_message = chat_storage.add_message(
        current_user.firebase_uid,
        chat_id,
        'assistant',
        response['explanation']
    )
    
    return jsonify({
        'response': response,
        'user_message_id': user_message['id'],
        'assistant_message_id': assistant_message['id'],
        'thread_id': thread_id
    })

@main.route('/observe_peer_attempt', methods=['POST'])
@login_required
def observe_peer_attempt():
    """Handle peer observation mode where student and teacher agents converse with streaming"""
    data = request.json
    topic = data.get('topic')
    chat_id = data.get('chat_id')
    num_exchanges = data.get('num_exchanges', 4)
    
    if not topic:
        return jsonify({'error': 'Topic is required'}), 400
    
    # Create new chat if not provided
    if not chat_id:
        chat = chat_storage.create_chat(
            current_user.firebase_uid,
            f"Peer Observation: {topic}"
        )
        chat_id = chat['id']
    else:
        chat = chat_storage.get_chat(current_user.firebase_uid, chat_id)
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404
    
    def generate_stream():
        """Generator function that yields SSE-formatted events"""
        try:
            # Send initial chat_id event
            yield f"data: {json.dumps({'type': 'chat_id', 'chat_id': chat_id, 'topic': topic})}\n\n"
            
            # Generate conversation and stream messages as they're created
            conversation_gen = peer_observation_handler.generate_conversation(
                topic=topic,
                num_exchanges=num_exchanges
            )
            
            for msg in conversation_gen:
                # Save message to chat storage immediately
                saved_message = chat_storage.add_message(
                    current_user.firebase_uid,
                    chat_id,
                    msg['role'],  # 'student' or 'teacher'
                    msg['content'],
                    'text'
                )
                
                # Send message event with saved message data
                yield f"data: {json.dumps({'type': 'message', 'message': saved_message})}\n\n"
            
            # Send completion event
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            # Send error event
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return Response(stream_with_context(generate_stream()), mimetype='text/event-stream') 