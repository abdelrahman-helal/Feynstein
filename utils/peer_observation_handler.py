import os
from typing import List, Dict, Any, Generator
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from utils.langchain_model_handler import FeynsteinModel

class PeerObservationHandler:
    """
    Handles peer observation mode where a student agent and teacher agent
    have a conversation about a physics topic.
    """
    
    # System prompts for student and teacher agents
    STUDENT_SYSTEM_PROMPT = """You are a physics student solving a problem.
CRITICAL STYLE RULES:
- Never say that you "made a mistake", "were wrong", or "realize an error"
- If your reasoning changes, treat it as clarification or continuation
- Use uncertainty language instead:
  - "Wait, maybe I should think about this differently"
  - "Let me try another approach"
- Speak naturally, like a real student thinking, not reflecting

BEHAVIOR:
- Reason step by step
- Include one conceptual misunderstanding common for this topic
- Include one small procedural slip
- Do NOT reach the correct final answer
- Stop after 3–4 reasoning steps"""

    TEACHER_SYSTEM_PROMPT = """You are a physics instructor responding to a student's reasoning.
You must:
- Identify the mistake explicitly
- Explain why it is incorrect
- Ask a guiding question instead of solving the problem
- Avoid giving the final answer
- End by encouraging independent problem solving

When starting a conversation, ask the student a question about the topic to get them thinking."""

    def __init__(self):
        """Initialize student and teacher models with their respective prompts"""
        # Create student model with student prompt
        self.student_model, self.student_prompt_template = FeynsteinModel.create_model_with_prompt(
            self.STUDENT_SYSTEM_PROMPT
        )
        
        # Create teacher model with teacher prompt
        self.teacher_model, self.teacher_prompt_template = FeynsteinModel.create_model_with_prompt(
            self.TEACHER_SYSTEM_PROMPT
        )
    
    def generate_conversation(self, topic: str, num_exchanges: int = 4) -> Generator[Dict[str, Any], None, None]:
        """
        Generate a conversation between student and teacher about a topic.
        Yields messages as they are created.
        
        Args:
            topic: The physics topic to discuss (e.g., "kinematics", "Newton's laws")
            num_exchanges: Number of student-teacher exchanges (default: 4)
            
        Yields:
            Dict with 'role' ('student' or 'teacher') and 'content' keys
        """
        student_messages = []
        teacher_messages = []
        
        # Initial prompt to teacher to ask a question about the topic
        initial_teacher_prompt = f"Ask the student a question about {topic} to get them thinking and solving a problem. Make it engaging and appropriate for a student to attempt."
        
        # Generate conversation loop
        for exchange in range(num_exchanges):
            # Teacher speaks first (or responds to student)
            if exchange == 0:
                # First exchange: Teacher asks initial question
                teacher_input = initial_teacher_prompt
            else:
                # Subsequent exchanges: Teacher responds to student's last message
                teacher_input = student_messages[-1].content if student_messages else initial_teacher_prompt
            
            # Invoke prompt template with teacher's message history and current input
            teacher_prompt = self.teacher_prompt_template.invoke({
                "messages": teacher_messages,  # Previous conversation history (without current input)
                "input": teacher_input
            })
            teacher_response = self.teacher_model.invoke(teacher_prompt)
            teacher_content = teacher_response.content if hasattr(teacher_response, 'content') else str(teacher_response)
            
            # Yield teacher message immediately
            yield {
                "role": "teacher",
                "content": teacher_content
            }
            
            # Add to teacher's message history for next turn
            teacher_messages.append(HumanMessage(content=teacher_input))
            teacher_messages.append(AIMessage(content=teacher_content))
            
            # Student responds to teacher
            # Use teacher's message as input
            student_input = teacher_content
            
            # Invoke prompt template with student's message history and teacher's question/response
            student_prompt = self.student_prompt_template.invoke({
                "messages": student_messages,  # Previous conversation history (without current input)
                "input": student_input
            })
            student_response = self.student_model.invoke(student_prompt)
            student_content = student_response.content if hasattr(student_response, 'content') else str(student_response)
            
            # Yield student message immediately
            yield {
                "role": "student",
                "content": student_content
            }
            
            # Add to student's message history for next turn
            student_messages.append(HumanMessage(content=student_input))
            student_messages.append(AIMessage(content=student_content))

# Global instance
peer_observation_handler = PeerObservationHandler()

