import os
import base64
from dotenv import load_dotenv
from utils.model_handler import physics_tutor
from utils.rag_system import PhysicsRAGSystem
from utils.equation_processor import EquationProcessor
from flask import Flask, render_template, request, jsonify

# Load environment variables
load_dotenv()

app = Flask(__name__)
equation_processor = EquationProcessor()  # Create an instance
physics_rag = PhysicsRAGSystem()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_question', methods=['POST'])
def process_question():
    data = request.json
    question_type = data.get('type')  # 'text' or 'image'
    content = data.get('content')
    
    if question_type == 'text':
        # Process text-based question
        context = physics_rag.query_relevant_content(query=content)
        response = physics_tutor.generate_step_by_step_guide(question=content, context=context)
        return jsonify(response)
    
    elif question_type == 'image':
        try:
            # Process image-based question
            image_base64 = content
            # Remove the data URL prefix if present
            if image_base64.startswith('data:image'):
                image_base64 = image_base64.split(',', 1)[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_base64)
            
            # Process the image using the instance
            equation, metadata = equation_processor.process_equation_image(image_bytes)
            context = PhysicsRAGSystem.query_relevant_content(equation)

            # Generate response with both the image and the extracted equation
            response = physics_tutor.generate_step_by_step_guide(
                question=f"Please help me understand this equation: {equation}",
                image_bytes=image_bytes,
                context=context
            )
            return jsonify(response)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return jsonify({
                "error": "Failed to process image",
                "details": str(e)
            }), 400
    
    return jsonify({'error': 'Invalid question type'}), 400

@app.route('/next_step', methods=['POST'])
def next_step():
    data = request.json
    # question = data.get('question')
    current_step = data.get('current_step', 0)
    student_response = data.get('response')
    image_base64 = data.get('image')  # Optionally support image in next steps
    image_bytes = None
    if image_base64:
        if image_base64.startswith('data:image'):
            image_base64 = image_base64.split(',', 1)[1]
        image_bytes = base64.b64decode(image_base64)
    # Generate next step based on student's response
    response = physics_tutor.generate_step_by_step_guide(
        current_step=current_step,
        student_response=student_response,
        image_bytes=image_bytes
    )
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True) 