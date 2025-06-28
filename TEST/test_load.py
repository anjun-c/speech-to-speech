import requests
import threading
import time
import json

BASE_URL = "http://localhost:8000"

def create_conversation_and_test():
    """Create a conversation and test TTS"""
    try:
        # Create conversation
        response = requests.post(f"{BASE_URL}/api/conversations/start")
        conversation_id = response.json()['conversation_id']
        
        # Test TTS
        tts_data = {"text": f"This is test message from thread {threading.current_thread().name}"}
        response = requests.post(
            f"{BASE_URL}/api/conversations/{conversation_id}/text-to-speech",
            json=tts_data
        )
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Thread {threading.current_thread().name} failed: {e}")
        return False

def load_test(num_threads=5):
    """Test with multiple concurrent requests"""
    print(f"🔥 Starting load test with {num_threads} threads...")
    
    threads = []
    results = []
    
    def worker():
        result = create_conversation_and_test()
        results.append(result)
    
    start_time = time.time()
    
    # Create and start threads
    for i in range(num_threads):
        thread = threading.Thread(target=worker, name=f"Thread-{i}")
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    
    success_count = sum(results)
    total_time = end_time - start_time
    
    print(f"✅ Load test completed:")
    print(f"   - Successful requests: {success_count}/{num_threads}")
    print(f"   - Total time: {total_time:.2f} seconds")
    print(f"   - Average time per request: {total_time/num_threads:.2f} seconds")
    
    return success_count == num_threads

if __name__ == "__main__":
    load_test()
