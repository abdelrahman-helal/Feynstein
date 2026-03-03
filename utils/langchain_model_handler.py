import os
from dotenv import load_dotenv
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

# Load environment variables
load_dotenv()

class State(TypedDict):
    messages: Annotated[List[BaseMessage], "add_messages"]
    context: Optional[str]

class FeynsteinModel:
    """
    Physics tutoring model using LangChain with support for multiple LLM providers.
    
    Supported providers:
    - groq: Fast free tier (default) - requires GROQ_API_KEY
    - huggingface/hf: Free inference API - requires HF_TOKEN
    - together: Free tier available - requires TOGETHER_API_KEY
    - ollama: Local models (completely free) - no API key needed
    - google_genai: Gemini models - requires GOOGLE_API_KEY
    
    Set LLM_PROVIDER environment variable to switch providers (default: 'groq')
    """
    def __init__(self):
        # Get provider from environment (default to 'groq' for free tier)
        # Set LLM_PROVIDER in .env to switch providers: groq, huggingface, together, ollama, google_genai
        self.provider = os.getenv("LLM_PROVIDER", "groq").lower()
        
        # Provider configuration mapping
        # Format: provider_name: (model_name, provider_string, api_key_env_var, api_key_value)
        provider_configs = {
            "groq": (
                "llama-3.1-8b-instant",  # Fast, free tier - 8B model
                # Alternative Groq models if this doesn't work:
                # "mixtral-8x7b-32768" - Mixtral model
                # "gemma-7b-it" - Google Gemma model
                "groq",
                "GROQ_API_KEY",
                os.getenv("GROQ_API_KEY")
            ),
            "huggingface": (
                "meta-llama/Llama-3.1-8B-Instruct",  # Free inference API
                "huggingface",
                "HF_TOKEN",
                os.getenv("HF_TOKEN")
            ),
            "hf": (
                "meta-llama/Llama-3.1-8B-Instruct",  # Alias for huggingface
                "huggingface",
                "HF_TOKEN",
                os.getenv("HF_TOKEN")
            ),
            "together": (
                "meta-llama/Llama-3-70b-chat-hf",  # Free tier available
                "together",
                "TOGETHER_API_KEY",
                os.getenv("TOGETHER_API_KEY")
            ),
            "ollama": (
                "llama3",  # Local model - requires Ollama installed locally
                "ollama",
                None,  # No API key needed for local Ollama
                None
            ),
            "google_genai": (
                "gemini-2.0-flash",  # Google Gemini (may have quota limits)
                "google_genai",
                "GOOGLE_API_KEY",
                os.getenv("GOOGLE_API_KEY")
            ),
        }
        
        # Get provider configuration
        if self.provider not in provider_configs:
            raise ValueError(
                f"Unsupported LLM provider: {self.provider}. "
                f"Supported providers: {', '.join(provider_configs.keys())}"
            )
        
        model_name, provider_string, api_key_env_var, api_key_value = provider_configs[self.provider]
        self.model_name = model_name
        self.provider_string = provider_string
        
        # Set API key if required (except for Ollama)
        if api_key_env_var and api_key_value:
            os.environ[api_key_env_var] = api_key_value
        elif api_key_env_var and not api_key_value:
            raise ValueError(
                f"API key not found for provider '{self.provider}'. "
                f"Please set {api_key_env_var} in your environment variables."
            )
        
        # Initialize the model
        print(f"Initializing {self.provider} model: {model_name}")
        try:
            self.model = init_chat_model(model_name, model_provider=provider_string)
            print(f"✓ Successfully initialized {self.provider} model")
        except Exception as e:
            error_msg = (
                f"Failed to initialize {self.provider} model: {str(e)}\n"
                f"Make sure you have:\n"
                f"1. Set {api_key_env_var} in your .env file (if required)\n"
                f"2. Installed required packages: pip install -r requirements.txt\n"
                f"3. For Ollama: Install and run Ollama locally (ollama pull {model_name})"
            )
            raise RuntimeError(error_msg) from e
        
        # System prompt for physics tutoring
        self.system_prompt = """You are Feynstein, an expert physics tutor named after Richard Feynman. 
        Your goal is to guide students through physics concepts step by step, making complex topics accessible.
        
        Key principles:
        1. Break down complex problems into simple, logical steps
        2. Use analogies and real-world examples when helpful
        3. Encourage critical thinking rather than just providing answers by asking follow up questions
        4.When using context from textbooks, incorporate it naturally into your explanations.
        5. Always maintain a conversational, encouraging tone unless the user assigns a specific tone
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
                    "model": self.model_name,
                    "provider": self.provider,
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
                    "model": self.model_name,
                    "provider": self.provider
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
    
    @classmethod
    def create_model_with_prompt(cls, system_prompt: str):
        """
        Create a model instance with a custom system prompt.
        Useful for creating student/teacher agents with different behaviors.
        
        Args:
            system_prompt: Custom system prompt for the model
            
        Returns:
            Tuple of (model, prompt_template) for use in conversations
        """
        # Get provider from environment (same as main model)
        provider = os.getenv("LLM_PROVIDER", "groq").lower()
        
        # Provider configuration mapping (same as __init__)
        provider_configs = {
            "groq": ("llama-3.1-8b-instant", "groq", "GROQ_API_KEY", os.getenv("GROQ_API_KEY")),
            "huggingface": ("meta-llama/Llama-3.1-8B-Instruct", "huggingface", "HF_TOKEN", os.getenv("HF_TOKEN")),
            "hf": ("meta-llama/Llama-3.1-8B-Instruct", "huggingface", "HF_TOKEN", os.getenv("HF_TOKEN")),
            "together": ("meta-llama/Llama-3-70b-chat-hf", "together", "TOGETHER_API_KEY", os.getenv("TOGETHER_API_KEY")),
            "ollama": ("llama3", "ollama", None, None),
            "google_genai": ("gemini-2.0-flash", "google_genai", "GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY")),
        }
        
        if provider not in provider_configs:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        model_name, provider_string, api_key_env_var, api_key_value = provider_configs[provider]
        
        # Set API key if required
        if api_key_env_var and api_key_value:
            os.environ[api_key_env_var] = api_key_value
        elif api_key_env_var and not api_key_value:
            raise ValueError(f"API key not found for provider '{provider}'. Please set {api_key_env_var} in your environment variables.")
        
        # Initialize the model
        model = init_chat_model(model_name, model_provider=provider_string)
        
        # Create prompt template with custom system prompt
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}")
        ])
        
        return model, prompt_template

# Create a singleton instance
feynstein_model = FeynsteinModel() 