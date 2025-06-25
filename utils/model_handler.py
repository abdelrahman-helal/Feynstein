import os
import base64
from typing import Dict, Any, Optional
from huggingface_hub import InferenceClient

class PhysicsTutorModel:
    def __init__(self):
        self.model_name = "deepseek-ai/DeepSeek-V3-0324"
        self.client = InferenceClient(
            provider="fireworks-ai",
            api_key=os.getenv("HF_TOKEN")
        )
        self.prompt = (
            "You are a physics tutor helping a student understand a concept.\n"
            "Your goal is to guide the student through the solution process step by step.\n"
            "Break down the solution into clear, logical steps. Don't reveal the entire solution at once.\n"
            "Focus on helping the student understand the underlying physics principles.\n\n"
        )

    def generate_step_by_step_guide(
        self,
        question: Optional[str] = None,
        current_step: int = 0,
        student_response: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a step-by-step guide for solving a physics problem using DeepSeek V3.
        Args:
            question: The question text
            current_step: Current step in the solution process
            student_response: Student's response to the previous step (optional)
            context: Optional context (e.g., textbook info)
            image_bytes: Optional image bytes for multimodal input
        Returns:
            Dict containing the response and metadata
        """
        # Build the step-by-step tutor prompt
        if question:
            self.prompt += (
                f"Question: {question}\n\n"
                f"Here is relevant information from physics textbooks that may help:\n{context if context else ''}\n\n"
                f"Current step: {current_step}\n"
            )
        if student_response:
            self.prompt += (
                f"\nStudent's response: {student_response}\n"
                f"Provide constructive feedback and guide to the next step. If the student's response shows misunderstanding, gently correct it and explain why. Use the textbook information to support your explanation. Do not include the same introductory answer from previous steps.\n"
                f"Current step: {current_step}\n"
            )
        else:
            self.prompt += (
                "\nStart with the first step of the solution. Explain the relevant physics principles that will be needed, referencing the textbook information when appropriate."
            )
        # Build multimodal content
        content = []
        # if image_bytes:
        #     # Convert bytes to base64 for API
        #     image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        #     content.append({"type": "image", "image": f"data:image/jpeg;base64,{image_base64}"})
        content.append({"type": "text", "text": self.prompt})
        print(f'content: {content}')
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )
            answer = completion.choices[0].message["content"] if hasattr(completion.choices[0], "message") else str(completion.choices[0])
            return {
                "explanation": answer.strip(),
                "next_steps": [],  # You may want to parse steps if needed
                "confidence": 0.8,
                "metadata": {
                    "model": self.model_name,
                    "context_used": bool(context),
                }
            }
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            return {
                "explanation": "I apologize, but I encountered an error processing your question. Please try again.",
                "next_steps": [],
                "confidence": 0.0,
                "metadata": {
                    "error": str(e),
                    "model": self.model_name
                }
            }

    def process_question(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        # For compatibility, just call generate_step_by_step_guide with default step and no student response
        return self.generate_step_by_step_guide(question, current_step=0, student_response=None, context=context)

# Create a singleton instance
physics_tutor = PhysicsTutorModel() 