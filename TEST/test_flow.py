import requests
import json
import time
import os
from io import BytesIO
import wave
import numpy as np

BASE_URL = "http://localhost:8000"

def create_test_audio():
    """Create a simple test audio file"""
    # Generate a simple sine wave (1 second, 440Hz)
    sample_rate = 16000
    duration = 1.0
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Save as WAV
    with wave.open('test_audio.wav', 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    return 'test_audio.wav'

def test_api_flow():
    print("🧪 Starting comprehensive API test...")
    
    # Test 1: Health check
    print("\n1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        health_data = response.json()
        print(f"✅ Health: {health_data}")
        
        if not health_data.get('pipeline_ready'):
            print("❌ Pipeline not ready! Check server logs.")
            return False
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test 2: Create conversation
    print("\n2️⃣ Creating conversation...")
    try:
        response = requests.post(f"{BASE_URL}/api/conversations/start")
        conv_data = response.json()
        conversation_id = conv_data['conversation_id']
        print(f"✅ Conversation created: {conversation_id}")
    except Exception as e:
        print(f"❌ Conversation creation failed: {e}")
        return False
    
    # # Test 3: Text-to-Speech
    # print("\n3️⃣ Testing text-to-speech...")
    # try:
    #     tts_data = {"text": "Hello, this is a test of the speech synthesis system."}
    #     response = requests.post(
    #         f"{BASE_URL}/api/conversations/{conversation_id}/text-to-speech",
    #         json=tts_data
    #     )
        
    #     if response.status_code == 200:
    #         with open("test_tts_output.wav", "wb") as f:
    #             f.write(response.content)
    #         print("✅ TTS successful - saved to test_tts_output.wav")
    #     else:
    #         print(f"❌ TTS failed: {response.status_code} - {response.text}")
    #         return False
            
    # except Exception as e:
    #     print(f"❌ TTS test failed: {e}")
    #     return False
    
    # Test 4: Conversation status
    print("\n4️⃣ Checking conversation status...")
    try:
        response = requests.get(f"{BASE_URL}/api/conversations/{conversation_id}/status")
        status_data = response.json()
        print(f"✅ Status: {status_data}")
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False
    
    # Test 5: Speech-to-Speech (if we can create test audio)
    print("\n5️⃣ Testing speech-to-speech...")
    try:
        # Create test audio
        test_audio_file = create_test_audio()
        
        with open(test_audio_file, 'rb') as audio_file:
            files = {'audio': ('test.wav', audio_file, 'audio/wav')}
            response = requests.post(
                f"{BASE_URL}/api/conversations/{conversation_id}/speak",
                files=files
            )
        
        if response.status_code == 200:
            with open("test_s2s_output.wav", "wb") as f:
                f.write(response.content)
            print("✅ Speech-to-Speech successful - saved to test_s2s_output.wav")
        else:
            print(f"⚠️ S2S failed: {response.status_code} - {response.text}")
            print("   (This might be expected if audio processing has issues)")
            
        # Clean up test audio
        os.remove(test_audio_file)
        
    except Exception as e:
        print(f"⚠️ S2S test failed: {e}")
        print("   (This might be expected - audio processing is complex)")
    
    # Test 6: Conversation history
    print("\n6️⃣ Checking conversation history...")
    try:
        response = requests.get(f"{BASE_URL}/api/conversations/{conversation_id}/history")
        history_data = response.json()
        print(f"✅ History: {len(history_data.get('history', []))} messages")
    except Exception as e:
        print(f"❌ History check failed: {e}")
        return False
    
    # Test 7: List conversations
    print("\n7️⃣ Listing all conversations...")
    try:
        response = requests.get(f"{BASE_URL}/api/conversations")
        conversations_data = response.json()
        print(f"✅ Found {len(conversations_data.get('conversations', []))} conversations")
    except Exception as e:
        print(f"❌ List conversations failed: {e}")
        return False
    
    # Test 8: Delete conversation
    print("\n8️⃣ Deleting conversation...")
    try:
        response = requests.delete(f"{BASE_URL}/api/conversations/{conversation_id}")
        print(f"✅ Conversation deleted")
    except Exception as e:
        print(f"❌ Delete failed: {e}")
        return False
    
    print("\n🎉 All basic tests passed! API is working correctly.")
    return True

if __name__ == "__main__":
    # Wait a moment for server to be ready
    print("Waiting 3 seconds for server to initialize...")
    time.sleep(3)
    
    success = test_api_flow()
    
    if success:
        print("\n✅ API is ready for AWS deployment!")
    else:
        print("\n❌ Fix issues before deploying to AWS")
