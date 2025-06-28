from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uuid
import asyncio
from typing import Dict, Optional, List
import base64
import io
import wave
import os
import tempfile
import threading
import time
from queue import Queue, Empty
import numpy as np

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Your existing imports
from conversation_manager import ConversationSession, ConversationManager
import sys
sys.path.append('..')
from s2s_pipeline import build_pipeline, parse_arguments, prepare_all_args, initialize_queues_and_events

# Pydantic models
class ConversationResponse(BaseModel):
    conversation_id: str
    status: str
    message: str

class ConversationStatus(BaseModel):
    conversation_id: str
    is_active: bool
    created_at: float
    last_activity: float
    message_count: int

class TextToSpeechRequest(BaseModel):
    text: str

class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    active_conversations: int

# Initialize FastAPI app
app = FastAPI(
    title="Speech-to-Speech API",
    description="API for conversational speech-to-speech interactions using DeepSeek",
    version="1.0.0"
)

# Global variables
conversation_manager = ConversationManager()
pipeline_manager = None
pipeline_ready = False

def create_pipeline_with_deepseek_config():
    """Initialize the pipeline with DeepSeek configuration"""
    global pipeline_ready
    
    try:
        # Set up arguments programmatically
        import sys
        original_argv = sys.argv.copy()
        
        sys.argv = [
            'api_server.py',
            '--llm', 'open_api',
            '--open_api_model_name', 'deepseek-chat',
            '--open_api_api_key', os.getenv('DEEPSEEK_API_KEY'),
            '--open_api_base_url', os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
            '--open_api_stream', 'false',  # Disable streaming for API simplicity
            '--mode', 'local',
            '--stt', 'faster-whisper',
            '--tts', 'parler',
            '--log_level', 'info'
        ]
        
        # Parse arguments and build pipeline
        args = parse_arguments()
        prepare_all_args(*args)
        queues_and_events = initialize_queues_and_events()
        pipeline = build_pipeline(*args, queues_and_events)
        
        # Restore original argv
        sys.argv = original_argv
        
        # Start the pipeline
        pipeline.start()
        pipeline_ready = True
        
        return pipeline, queues_and_events
        
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        pipeline_ready = False
        raise

# Initialize pipeline on startup
@app.on_event("startup")
async def startup_event():
    global pipeline_manager, pipeline_queues
    try:
        pipeline_manager, pipeline_queues = create_pipeline_with_deepseek_config()
        print("Pipeline initialized successfully")
    except Exception as e:
        print(f"Failed to start pipeline: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    global pipeline_manager
    if pipeline_manager:
        pipeline_manager.stop()

# Helper function to process audio through pipeline
async def process_audio_through_pipeline(audio_data: bytes, session_id: str) -> bytes:
    """Process audio through the speech-to-speech pipeline"""
    global pipeline_queues
    
    if not pipeline_ready or not pipeline_queues:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    
    try:
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name
        
        # Convert audio to the format expected by the pipeline
        # This is a simplified version - you may need to adjust based on your audio format requirements
        
        # Put audio chunks into the pipeline
        recv_queue = pipeline_queues["recv_audio_chunks_queue"]
        send_queue = pipeline_queues["send_audio_chunks_queue"]
        
        # Read audio file and put chunks into queue
        with wave.open(temp_audio_path, 'rb') as wav_file:
            chunk_size = 1024
            while True:
                chunk = wav_file.readframes(chunk_size)
                if not chunk:
                    break
                recv_queue.put(chunk)
        
        # Wait for response audio
        response_chunks = []
        timeout = 30  # 30 second timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                chunk = send_queue.get(timeout=1)
                if chunk == b"END":
                    break
                response_chunks.append(chunk)
            except Empty:
                continue
        
        # Combine response chunks
        response_audio = b''.join(response_chunks)
        
        # Clean up
        os.unlink(temp_audio_path)
        
        return response_audio
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

# API Routes

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if pipeline_ready else "unhealthy",
        pipeline_ready=pipeline_ready,
        active_conversations=len(conversation_manager.sessions)
    )

@app.post("/api/conversations/start", response_model=ConversationResponse)
async def start_conversation():
    """Start a new conversation session"""
    if not pipeline_ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    
    session_id = conversation_manager.create_session()
    return ConversationResponse(
        conversation_id=session_id,
        status="created",
        message="Conversation started successfully"
    )

@app.get("/api/conversations/{conversation_id}/status", response_model=ConversationStatus)
async def get_conversation_status(conversation_id: str):
    """Get conversation status"""
    session = conversation_manager.get_session(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return ConversationStatus(
        conversation_id=conversation_id,
        is_active=session.is_active,
        created_at=session.created_at,
        last_activity=session.last_activity,
        message_count=len(session.conversation_history)
    )

@app.post("/api/conversations/{conversation_id}/speak")
async def speak_to_conversation(
    conversation_id: str,
    audio: UploadFile = File(..., description="Audio file (WAV format preferred)")
):
    """Send audio to conversation and get audio response"""
    session = conversation_manager.get_session(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if not audio.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    try:
        # Read audio data
        audio_data = await audio.read()
        
        # Process through pipeline
        response_audio = await process_audio_through_pipeline(audio_data, conversation_id)
        
        # Save response to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_response:
            temp_response.write(response_audio)
            temp_response_path = temp_response.name
        
        # Return audio file
        return FileResponse(
            temp_response_path,
            media_type='audio/wav',
            filename=f"response_{conversation_id}.wav"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/api/conversations/{conversation_id}/text-to-speech")
async def text_to_speech(
    conversation_id: str,
    request: TextToSpeechRequest
):
    """Convert text to speech for a conversation"""
    session = conversation_manager.get_session(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    try:
        # Put text directly into LM response queue to skip STT
        lm_response_queue = pipeline_queues["lm_response_queue"]
        lm_response_queue.put(request.text)
        
        # Wait for TTS response
        send_queue = pipeline_queues["send_audio_chunks_queue"]
        response_chunks = []
        timeout = 30
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                chunk = send_queue.get(timeout=1)
                if chunk == b"END":
                    break
                response_chunks.append(chunk)
            except Empty:
                continue
        
        response_audio = b''.join(response_chunks)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_response:
            temp_response.write(response_audio)
            temp_response_path = temp_response.name
        
        # Add to conversation history
        session.add_to_history("assistant", request.text)
        
        return FileResponse(
            temp_response_path,
            media_type='audio/wav',
            filename=f"tts_{conversation_id}.wav"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

@app.get("/api/conversations/{conversation_id}/history")
async def get_conversation_history(conversation_id: str):
    """Get conversation history"""
    session = conversation_manager.get_session(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "history": session.get_history()
    }

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation session"""
    success = conversation_manager.delete_session(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"message": "Conversation deleted successfully"}

@app.get("/api/conversations")
async def list_conversations():
    """List all active conversations"""
    conversations = []
    for session_id, session in conversation_manager.sessions.items():
        conversations.append({
            "conversation_id": session_id,
            "created_at": session.created_at,
            "last_activity": session.last_activity,
            "message_count": len(session.conversation_history),
            "is_active": session.is_active
        })
    
    return {"conversations": conversations}

# Cleanup endpoint
@app.post("/api/admin/cleanup")
async def cleanup_expired_conversations():
    """Clean up expired conversations"""
    cleaned = conversation_manager.cleanup_expired_sessions()
    return {"message": f"Cleaned up {cleaned} expired conversations"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
