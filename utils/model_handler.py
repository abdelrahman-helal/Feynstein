from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Any
from .rag_system import physics_rag

class PhysicsTutorModel:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            load_in_8bit=True  # Enable 8-bit quantization for memory efficiency
        )
        
    def generate_step_by_step_guide(self, question: str, current_step: int = 0, 
                                  student_response: str = None) -> Dict[str, Any]:
        """
        Generate a step-by-step guide for solving a physics problem.
        """
        # Get relevant textbook content
        textbook_context = physics_rag.get_context_for_question(question)
        
        # Create a prompt that encourages step-by-step thinking
        prompt = self._create_prompt(question, current_step, student_response, textbook_context)
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=1024,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_response(response)
    
    def _create_prompt(self, question: str, current_step: int, 
                      student_response: str = None, textbook_context: str = None) -> str:
        """
        Create a prompt that guides the model to provide step-by-step explanations.
        """
        base_prompt = f"""<s>[INST] You are a physics tutor helping a student understand a concept. 
        Your goal is to guide the student through the solution process step by step.
        Break down the solution into clear, logical steps. Don't reveal the entire solution at once.
        Focus on helping the student understand the underlying physics principles.
        
        Here is relevant information from physics textbooks that may help:
        {textbook_context}
        
        Question: {question}
        
        Current step: {current_step}
        """
        
        if student_response:
            base_prompt += f"\nStudent's response: {student_response}\n"
            base_prompt += "Provide constructive feedback and guide to the next step. If the student's response shows misunderstanding, gently correct it and explain why. Use the textbook information to support your explanation."
        else:
            base_prompt += "\nStart with the first step of the solution. Explain the relevant physics principles that will be needed, referencing the textbook information when appropriate."
            
        base_prompt += " [/INST]"
        return base_prompt
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the model's response into a structured format.
        """
        # Split response into steps
        steps = response.split('\n')
        steps = [step.strip() for step in steps if step.strip()]
        
        return {
            'explanation': steps[0] if steps else "",
            'next_steps': steps[1:] if len(steps) > 1 else [],
            'hints': self._extract_hints(steps),
            'principles': self._extract_physics_principles(steps)
        }
    
    def _extract_hints(self, steps: List[str]) -> List[str]:
        """
        Extract hints from the steps to help guide the student.
        """
        hints = []
        for step in steps:
            if "hint" in step.lower() or "consider" in step.lower() or "remember" in step.lower():
                hints.append(step)
        return hints
    
    def _extract_physics_principles(self, steps: List[str]) -> List[str]:
        """
        Extract key physics principles mentioned in the steps.
        """
        principles = []
        physics_keywords = [
            "conservation", "force", "energy", "momentum", "velocity",
            "acceleration", "mass", "gravity", "electric", "magnetic",
            "wave", "particle", "field", "potential", "kinetic",
            "quantum", "classical", "relativity", "hamiltonian", "lagrangian"
        ]
        
        for step in steps:
            for keyword in physics_keywords:
                if keyword in step.lower():
                    principles.append(step)
                    break
        return principles

# Create a singleton instance
physics_tutor = PhysicsTutorModel() 