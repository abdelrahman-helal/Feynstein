import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid

class ChatStorage:
    """Simple chat storage using JSON files instead of SQLAlchemy"""
    
    def __init__(self, storage_dir: str = "chat_data"):
        self.storage_dir = storage_dir
        self._ensure_storage_dir()
    
    def _ensure_storage_dir(self):
        """Ensure the storage directory exists"""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
    
    def _get_user_file(self, firebase_uid: str) -> str:
        """Get the file path for a user's chat data"""
        return os.path.join(self.storage_dir, f"{firebase_uid}.json")
    
    def _load_user_data(self, firebase_uid: str) -> Dict[str, Any]:
        """Load user's chat data from file"""
        file_path = self._get_user_file(firebase_uid)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading user data: {e}")
        return {"chats": []}
    
    def _save_user_data(self, firebase_uid: str, data: Dict[str, Any]):
        """Save user's chat data to file"""
        file_path = self._get_user_file(firebase_uid)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"Error saving user data: {e}")
    
    def get_user_chats(self, firebase_uid: str) -> List[Dict[str, Any]]:
        """Get all chats for a user"""
        data = self._load_user_data(firebase_uid)
        return data.get("chats", [])
    
    def get_chat(self, firebase_uid: str, chat_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chat by ID"""
        chats = self.get_user_chats(firebase_uid)
        for chat in chats:
            if chat.get("id") == chat_id:
                return chat
        return None
    
    def create_chat(self, firebase_uid: str, title: str) -> Dict[str, Any]:
        """Create a new chat"""
        data = self._load_user_data(firebase_uid)
        chats = data.get("chats", [])
        
        new_chat = {
            "id": str(uuid.uuid4()),
            "title": title,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "messages": []
        }
        
        chats.append(new_chat)
        data["chats"] = chats
        self._save_user_data(firebase_uid, data)
        
        return new_chat
    
    def add_message(self, firebase_uid: str, chat_id: str, role: str, content: str, question_type: str = "text", context_used: bool = False) -> Dict[str, Any]:
        """Add a message to a chat"""
        data = self._load_user_data(firebase_uid)
        chats = data.get("chats", [])
        
        for chat in chats:
            if chat.get("id") == chat_id:
                message = {
                    "id": str(uuid.uuid4()),
                    "role": role,
                    "content": content,
                    "timestamp": datetime.utcnow().isoformat(),
                    "question_type": question_type,
                    "context_used": context_used
                }
                
                chat["messages"].append(message)
                chat["updated_at"] = datetime.utcnow().isoformat()
                
                self._save_user_data(firebase_uid, data)
                return message
        
        raise ValueError(f"Chat with ID {chat_id} not found")
    
    def delete_chat(self, firebase_uid: str, chat_id: str) -> bool:
        """Delete a chat"""
        data = self._load_user_data(firebase_uid)
        chats = data.get("chats", [])
        
        for i, chat in enumerate(chats):
            if chat.get("id") == chat_id:
                del chats[i]
                data["chats"] = chats
                self._save_user_data(firebase_uid, data)
                return True
        
        return False
    
    def update_chat_title(self, firebase_uid: str, chat_id: str, title: str) -> bool:
        """Update chat title"""
        data = self._load_user_data(firebase_uid)
        chats = data.get("chats", [])
        
        for chat in chats:
            if chat.get("id") == chat_id:
                chat["title"] = title
                chat["updated_at"] = datetime.utcnow().isoformat()
                self._save_user_data(firebase_uid, data)
                return True
        
        return False

# Global chat storage instance
chat_storage = ChatStorage() 