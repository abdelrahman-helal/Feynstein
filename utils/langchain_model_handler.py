import os
import getpass
from typing import Dict, Any, Optional, List
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import trim_messages
from typing_extensions import Annotated, TypedDict
import json

class State(TypedDict):
    messages: Annotated[List[BaseMessage], "add_messages"]
    context: Optional[str]

class FeynsteinModel:
    def __init__(self):
        # Initialize Gemini model using init_chat_model
        # if not os.environ.get("OPENAI_API_KEY"):
        #     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
        self.model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
        
        # System prompt for physics tutoring
        self.system_prompt = """You are Feynstein, an expert physics tutor named after Richard Feynman. 
        Your goal is to guide students through physics concepts step by step, making complex topics accessible.
        
        Key principles:
        1. Break down complex problems into simple, logical steps
        2. Use analogies and real-world examples when helpful
        3. Encourage critical thinking rather than just providing answers by asking follow up questions
        4. When using context from textbooks, incorporate it naturally into your explanations.
        5. Always maintain a conversational, encouraging tone unless the user assigns a specific tone
        6. Write any equations in LaTeX format ($$ or $)
        """
        
        # Create the prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "Context from physics textbooks: {context}\n\nStudent's question: {input}")
        ])
        
        # Create message trimmer
        self.trimmer = trim_messages(
            max_tokens=4000,  # Adjust based on model context window
            strategy="last",
            token_counter=self.model,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )
        
        # Create the workflow
        self.workflow = self._create_workflow()
        
        # Initialize memory saver for checkpoints
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for the chatbot"""
        
        def call_model(state: State):
            # Trim messages to prevent context overflow
            trimmed_messages = self.trimmer.invoke(state["messages"])
            
            # Get the last human message
            last_message = None
            for msg in reversed(trimmed_messages):
                if isinstance(msg, HumanMessage):
                    last_message = msg
                    break
            
            if not last_message:
                return {"messages": [AIMessage(content="I didn't receive a question. Could you please ask your physics question?")]}
            
            # Create the prompt with context
            context = state.get("context", "")
            prompt = self.prompt_template.invoke({
                "messages": trimmed_messages[:-1],  # Exclude the last human message
                "context": context,
                "input": last_message.content
            })
            
            # Get response from model
            response = self.model.invoke(prompt)
            return {"messages": [response]}
        
        # Create the graph
        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_node("model", call_model)
        workflow.add_edge(START, "model")
        workflow.add_edge("model", END)
        
        return workflow
    
    def generate_response(
        self,
        question: str,
        thread_id: str,
        context: Optional[str] = None,
        question_type: str = "text"
    ) -> Dict[str, Any]:
        """
        Generate a response using the LangChain workflow with message persistence
        
        Args:
            question: The student's question
            thread_id: Unique identifier for the conversation thread
            context: Optional context from RAG system
            question_type: Type of question ('text' or 'image')
        
        Returns:
            Dict containing the response and metadata
        """
        try:
            # Create human message
            human_message = HumanMessage(content=question)
            
            # Prepare state
            state = {
                "messages": [human_message],
                "context": context or ""
            }
            
            # Configuration for thread persistence
            config = {"configurable": {"thread_id": thread_id}}
            
            # Get response from the workflow
            result = self.app.invoke(state, config)
            # Extract the last AI message
            ai_message = None
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    ai_message = msg
                    break
            
            if not ai_message:
                return {
                    "explanation": "I apologize, but I encountered an error processing your question.",
                    "next_steps": [],
                    "confidence": 0.0,
                    "metadata": {"error": "No AI response generated"}
                }
            
            return {
                "explanation": ai_message.content,
                "next_steps": [],
                "metadata": {
                    "model": "gemini-2.0-flash",
                    "context_used": bool(context),
                    "thread_id": thread_id,
                    "question_type": question_type
                }
            }
            
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            return {
                "explanation": f"I apologize, but I encountered an error: {str(e)}",
                "next_steps": [],
                "metadata": {
                    "error": str(e),
                    "model": "gemini-2.0-flash"
                }
            }
    
    def get_conversation_history(self, thread_id: str) -> List[BaseMessage]:
        """Get the conversation history for a specific thread"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            # This would need to be implemented based on how you want to retrieve history
            # For now, we'll return an empty list
            return []
        except Exception as e:
            print(f"Error getting conversation history: {str(e)}")
            return []

# Create a singleton instance
feynstein_model = FeynsteinModel() 