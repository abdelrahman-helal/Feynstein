# AI Physics Tutor

An intelligent physics tutoring system that helps students understand physics concepts through step-by-step derivations and explanations.

## Features

- Interactive web interface for student interaction
- Image-based equation recognition
- Text-based question processing
- Step-by-step derivation guidance
- Adaptive learning approach

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_api_key_here
```

3. Run the application:
```bash
python app.py
```

## Project Structure

- `app.py`: Main Flask application
- `static/`: Static files (CSS, JavaScript, images)
- `templates/`: HTML templates
- `utils/`: Utility functions
  - `image_processing.py`: Image and equation recognition
  - `physics_engine.py`: Physics concept processing
  - `tutoring_logic.py`: Step-by-step tutoring logic

## Contributing

Feel free to submit issues and enhancement requests! 
