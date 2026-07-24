import os
import requests

def test_api():
    url = "http://localhost:8000/detect"
    
    # We will test using one of the ESC-50 real audio files we already downloaded
    test_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "processed", "siren", "siren_esc50_000.wav"))
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        # Try a different one
        test_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "processed", "glass_breaking", "glass_breaking_esc50_000.wav"))
        if not os.path.exists(test_file):
            print("No test files available yet.")
            return
            
    print(f"Testing API with real audio file: {test_file}")
    
    with open(test_file, "rb") as f:
        files = {"file": ("test.wav", f, "audio/wav")}
        data = {
            "duration": 5.0, # Send 5s to trigger both Pass 1 and Pass 2
            "media_playback": False,
            "sudden_motion": False
        }
        
        try:
            response = requests.post(url, files=files, data=data)
            print(f"Status Code: {response.status_code}")
            import json
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"Failed to connect to API: {e}")

if __name__ == "__main__":
    test_api()
