from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from utils.image_processing import process_image
from utils.model_handler import physics_tutor

# Load environment variables
load_dotenv()

app = Flask(__name__)

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
        response = physics_tutor.generate_step_by_step_guide(content)
        return jsonify(response)
    
    elif question_type == 'image':
        # Process image-based question
        image_data = content
        equation = process_image(image_data)
        response = physics_tutor.generate_step_by_step_guide(equation)
        return jsonify(response)
    
    return jsonify({'error': 'Invalid question type'}), 400

@app.route('/next_step', methods=['POST'])
def next_step():
    data = request.json
    question = data.get('question')
    current_step = data.get('current_step', 0)
    student_response = data.get('response')
    
    # Generate next step based on student's response
    response = physics_tutor.generate_step_by_step_guide(
        question,
        current_step=current_step,
        student_response=student_response
    )
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True) 