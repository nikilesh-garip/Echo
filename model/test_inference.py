import os
import requests

def test_api():
    url = "http://localhost:8000/detect"
    
    # Find first siren or glass breaking file dynamically
    import glob
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "processed"))
    
    test_files = glob.glob(os.path.join(processed_dir, "siren", "*.wav"))
    if not test_files:
        test_files = glob.glob(os.path.join(processed_dir, "glass_breaking", "*.wav"))
        
    if not test_files:
        print("No processed test files available yet.")
        return
        
    test_file = test_files[0]
            
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
