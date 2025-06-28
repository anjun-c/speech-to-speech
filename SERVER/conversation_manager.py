from typing import Dict, List, Optional
import uuid
import time
import asyncio
from queue import Queue
import threading
import tempfile
import os
import wave
import numpy as np

class ConversationSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conversation_history = []
        self.created_at = time.time()
        self.last_activity = time.time()
        self.is_active = True
        
    def add_to_history(self, role: str, content: str):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        self.last_activity = time.time()
        
    def get_history(self):
        return self.conversation_history
        
    def is_expired(self, timeout_minutes=30):
        return (time.time() - self.last_activity) > (timeout_minutes * 60)

class ConversationManager:
    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}
        
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        session = ConversationSession(session_id)
        self.sessions[session_id] = session
        return session_id
        
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        return self.sessions.get(session_id)
        
    def delete_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
        
    def cleanup_expired_sessions(self):
        expired_sessions = [
            sid for sid, session in self.sessions.items() 
            if session.is_expired()
        ]
        for sid in expired_sessions:
            del self.sessions[sid]
        return len(expired_sessions)
