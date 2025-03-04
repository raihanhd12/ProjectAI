import os
import json
import uuid
import datetime

# Chat history file path
CHAT_HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chat_history.json")


class ChatHistory:
    @staticmethod
    def load_history():
        """
        Load chat history from file.
        
        Returns:
            Dictionary containing conversation history
        """
        if os.path.exists(CHAT_HISTORY_FILE):
            try:
                with open(CHAT_HISTORY_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    @staticmethod
    def save_history(history):
        """
        Save chat history to file.
        
        Args:
            history: Dictionary containing conversation history
        """
        with open(CHAT_HISTORY_FILE, "w") as f:
            json.dump(history, f)
    
    @staticmethod
    def add_conversation(title=None, app_mode="RAG Chat"):
        """
        Add a new conversation to the history.
        
        Args:
            title: Optional title for the conversation
            app_mode: The application mode for this conversation
            
        Returns:
            The new conversation ID
        """
        history = ChatHistory.load_history()
        conversation_id = str(uuid.uuid4())
        
        if not title:
            title = f"Conversation {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        history[conversation_id] = {
            "title": title,
            "messages": [],
            "timestamp": datetime.datetime.now().isoformat(),
            "app_mode": app_mode
        }
        
        ChatHistory.save_history(history)
        return conversation_id
    
    @staticmethod
    def add_message(conversation_id, role, content, metadata=None):
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: The conversation ID
            role: The role of the message sender ("user" or "assistant")
            content: The message content
            metadata: Optional metadata for the message
        """
        history = ChatHistory.load_history()
        
        if conversation_id not in history:
            conversation_id = ChatHistory.add_conversation()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        if metadata:
            message.update(metadata)
        
        history[conversation_id]["messages"].append(message)
        
        ChatHistory.save_history(history)
    
    @staticmethod
    def get_messages(conversation_id):
        """
        Get all messages for a conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            List of messages for the conversation
        """
        history = ChatHistory.load_history()
        
        if conversation_id in history:
            return history[conversation_id]["messages"]
        
        return []
    
    @staticmethod
    def delete_conversation(conversation_id):
        """
        Delete a conversation from history.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            True if conversation was deleted, False otherwise
        """
        history = ChatHistory.load_history()
        
        if conversation_id in history:
            del history[conversation_id]
            ChatHistory.save_history(history)
            return True
        
        return False
    
    @staticmethod
    def update_message(conversation_id, message_index, updated_content):
        """
        Update an existing message in a conversation.
        
        Args:
            conversation_id: The conversation ID
            message_index: The index of the message to update
            updated_content: The new content for the message
            
        Returns:
            True if message was updated, False otherwise
        """
        history = ChatHistory.load_history()
        
        if conversation_id in history:
            if 0 <= message_index < len(history[conversation_id]["messages"]):
                history[conversation_id]["messages"][message_index]["content"] = updated_content
                ChatHistory.save_history(history)
                return True
        
        return False