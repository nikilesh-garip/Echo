import os
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_endpoints():
    print("Testing FastAPI Backend Endpoints...")
    
    # 1. Test POST /events (Log a detection event)
    event_data = {
        "user_id": "test_user_123",
        "class_name": "gunshot",
        "primary_conf": 0.85,
        "verification_conf": 0.80,
        "risk_score": 75,
        "risk_level": "POSSIBLE_DANGER"
    }
    
    response = client.post("/events", json=event_data)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    res_data = response.json()
    assert res_data["class_name"] == "gunshot"
    assert res_data["risk_level"] == "POSSIBLE_DANGER"
    assert "timestamp" in res_data
    event_id = res_data["id"]
    print(f"-> Event Logged Successfully: ID {event_id}")

    # 2. Test GET /events/{user_id} (History retrieval)
    response = client.get("/events/test_user_123")
    assert response.status_code == 200
    history = response.json()
    assert len(history) >= 1
    assert history[0]["class_name"] == "gunshot"
    print(f"-> History Retrieved Successfully. Found {len(history)} records.")

    # 3. Test POST /contacts (Add emergency contact)
    contact_data = {
        "user_id": "test_user_123",
        "name": "John Doe",
        "phone": "+123456789",
        "relation": "Friend"
    }
    response = client.post("/contacts", json=contact_data)
    assert response.status_code == 200
    res_contact = response.json()
    assert res_contact["name"] == "John Doe"
    contact_id = res_contact["id"]
    print(f"-> Contact Added Successfully: ID {contact_id}")

    # 4. Test GET /contacts/{user_id} (Retrieve contacts)
    response = client.get("/contacts/test_user_123")
    assert response.status_code == 200
    contacts = response.json()
    assert len(contacts) >= 1
    assert contacts[0]["name"] == "John Doe"
    print(f"-> Contacts Retrieved Successfully. Found {len(contacts)} contacts.")

    # 5. Test DELETE /contacts/{contact_id}
    response = client.delete(f"/contacts/{contact_id}")
    assert response.status_code == 200
    print("-> Contact Deleted Successfully.")

    # Verify deleted
    response = client.get("/contacts/test_user_123")
    assert response.status_code == 200
    contacts_after = response.json()
    assert len(contacts_after) == 0, f"Expected 0 contacts, got {len(contacts_after)}"

    # 6. Test GET /nearby (OSM Proxy/Fallback)
    response = client.get("/nearby?lat=37.7749&lng=-122.4194&type=hospital")
    assert response.status_code == 200
    nearby_data = response.json()
    assert "results" in nearby_data
    assert len(nearby_data["results"]) >= 1
    print(f"-> Nearby Hospital Lookup Successful. Found {len(nearby_data['results'])} results.")

    # 7. Test POST /demo/nearby-corroboration
    response = client.post("/demo/nearby-corroboration?lat=37.7749&lng=-122.4194&class_name=gunshot")
    assert response.status_code == 200
    corrob_data = response.json()
    assert corrob_data["simulated"] is True
    assert corrob_data["alert_corroborated"] is True
    print("-> Simulated Corroboration Endpoint Successful.")

    print("\nAll Backend Endpoint Tests Passed Successfully!")

if __name__ == "__main__":
    test_endpoints()
