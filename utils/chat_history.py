import json
import os
import uuid
from datetime import datetime


class ChatHistory:
    """
    Utility class to manage chat history persistently across sessions.
    """
    HISTORY_FILE = "chat_history.json"

    @classmethod
    def load_history(cls):
        """Load chat history from file or create empty history"""
        if os.path.exists(cls.HISTORY_FILE):
            try:
                with open(cls.HISTORY_FILE, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading chat history: {e}")
                return {}
        return {}

    @classmethod
    def save_history(cls, history):
        """Save chat history to file"""
        try:
            with open(cls.HISTORY_FILE, "w") as f:
                json.dump(history, f, indent=2)
        except IOError as e:
            print(f"Error saving chat history: {e}")

    @classmethod
    def add_conversation(cls, title=None, app_mode="rag_chat"):
        """
        Create a new conversation entry in history.

        Args:
            title: Optional title for the conversation
            app_mode: Application mode for this conversation

        Returns:
            String ID of the new conversation
        """
        history = cls.load_history()

        # Generate a unique ID
        conversation_id = str(uuid.uuid4())

        # Create default title if none provided
        if not title:
            current_time = datetime.now().strftime("%b %d, %Y %I:%M %p")
            title = f"Conversation {current_time}"

        # Add conversation to history
        history[conversation_id] = {
            "title": title,
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "app_mode": app_mode
        }

        cls.save_history(history)
        return conversation_id

    @classmethod
    def delete_conversation(cls, conversation_id):
        """Delete a conversation from history"""
        history = cls.load_history()
        if conversation_id in history:
            del history[conversation_id]
            cls.save_history(history)
            return True
        return False

    @classmethod
    def get_messages(cls, conversation_id):
        """Get all messages for a specific conversation"""
        history = cls.load_history()
        if conversation_id in history:
            return history[conversation_id].get("messages", [])
        return []

    @classmethod
    def add_message(cls, conversation_id, role, content, metadata=None):
        """
        Add a message to a conversation.

        Args:
            conversation_id: ID of the conversation
            role: Role of the message sender ("user" or "assistant")
            content: Content of the message
            metadata: Optional metadata for the message

        Returns:
            Boolean indicating success or failure
        """
        history = cls.load_history()

        if conversation_id not in history:
            return False

        # Create message object
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        # Add metadata if provided
        if metadata:
            for key, value in metadata.items():
                message[key] = value

        # Add to messages list
        history[conversation_id]["messages"].append(message)

        # Update title to first user message if it's a default title
        if (role == "user" and
            history[conversation_id]["title"].startswith("Conversation ") and
                len(history[conversation_id]["messages"]) <= 2):
            # Truncate long messages
            if len(content) > 40:
                title = content[:37] + "..."
            else:
                title = content
            history[conversation_id]["title"] = title

        cls.save_history(history)
        return True
