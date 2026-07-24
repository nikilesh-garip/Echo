import os
import sys
import time
import sqlite3
import requests
# Force uvicorn reload after training
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add parent directory to path so we can import model modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model")))

from two_pass_detector import TwoPassDetector
from risk_scorer import RiskScorer

DB_PATH = "echo_backend.db"
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "checkpoints", "best_model.pth"))

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            class_name TEXT NOT NULL,
            primary_conf REAL NOT NULL,
            verification_conf REAL NOT NULL,
            risk_score INTEGER NOT NULL,
            risk_level TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            phone TEXT NOT NULL,
            relation TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

app = FastAPI(title="Echo Smart Emergency System API", version="1.0")

# Initialize Detector and Scorer
detector = None
if os.path.exists(MODEL_PATH):
    try:
        detector = TwoPassDetector(MODEL_PATH)
        print("Successfully loaded CRNN model for inference!")
    except Exception as e:
        print(f"Error loading model: {e}")

scorer = RiskScorer()

# Pydantic Schemas
class EventCreate(BaseModel):
    user_id: str
    class_name: str
    primary_conf: float
    verification_conf: float
    risk_score: int
    risk_level: str

class EventResponse(BaseModel):
    id: int
    user_id: str
    timestamp: float
    class_name: str
    primary_conf: float
    verification_conf: float
    risk_score: int
    risk_level: str

class ContactCreate(BaseModel):
    user_id: str
    name: str
    phone: str
    relation: Optional[str] = None

class ContactResponse(BaseModel):
    id: int
    user_id: str
    name: str
    phone: str
    relation: Optional[str] = None


@app.post("/detect")
async def detect_audio(
    file: UploadFile = File(...),
    duration: float = Form(..., description="Duration of the audio clip in seconds"),
    media_playback: bool = Form(False, description="Whether media is playing on device"),
    sudden_motion: bool = Form(False, description="Whether sudden motion is active")
):
    """
    Performs real-time two-pass detection and risk scoring on an uploaded audio clip.
    If duration is ~2.0s, performs Pass 1 (Primary).
    If duration is ~5.0s, performs Pass 2 (Verification).
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="CRNN model is not loaded on backend.")
        
    try:
        # Read uploaded file bytes
        file_bytes = await file.read()
        
        # Save temp file to read with soundfile
        os.makedirs("temp", exist_ok=True)
        temp_path = f"temp/{time.time()}_chunk.wav"
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
            
        import soundfile as sf
        audio_data, sr = sf.read(temp_path)
        os.remove(temp_path) # Clean up
        
        if duration <= 3.0:
            # Run Pass 1
            has_candidate, candidate, confidence = detector.run_pass_1(audio_data, sr)
            
            # Immediate verification for transient events (gunshot, explosion, glass_breaking)
            if has_candidate and candidate in ["gunshot", "explosion", "glass_breaking"]:
                risk_score, risk_level = scorer.calculate_risk(
                    primary_conf=confidence,
                    verification_conf=confidence,
                    media_playback=media_playback,
                    sudden_motion=sudden_motion,
                    current_class=candidate
                )
                return {
                    "pass": 1,
                    "has_candidate": True,
                    "candidate": candidate,
                    "confidence": confidence,
                    "immediate_verification": True,
                    "verified": True,
                    "primary_confidence": confidence,
                    "verification_confidence": confidence,
                    "risk_score": risk_score,
                    "risk_level": risk_level
                }
                
            return {
                "pass": 1,
                "has_candidate": has_candidate,
                "candidate": candidate,
                "confidence": confidence,
                "immediate_verification": False
            }
        else:
            # Run Pass 2 (Requires candidate parameter to be verified)
            # Pass 2 is triggered by client with a 5s buffer. We evaluate the candidate.
            # For simplicity, we search for the highest non-normal class in Pass 2 as verification
            has_candidate, candidate, p1_conf = detector.run_pass_1(audio_data[:int(len(audio_data)*2/5)], sr)
            
            if not has_candidate:
                return {
                    "pass": 2,
                    "verified": False,
                    "candidate": "normal",
                    "confidence": p1_conf,
                    "risk_score": 0,
                    "risk_level": "NORMAL"
                }
                
            verified, p2_conf = detector.run_pass_2(audio_data, sr, candidate)
            
            # Calculate Risk Score
            risk_score, risk_level = scorer.calculate_risk(
                primary_conf=p1_conf,
                verification_conf=p2_conf if verified else 0.0,
                media_playback=media_playback,
                sudden_motion=sudden_motion,
                current_class=candidate if verified else "normal"
            )
            
            return {
                "pass": 2,
                "verified": verified,
                "candidate": candidate,
                "primary_confidence": p1_conf,
                "verification_confidence": p2_conf,
                "risk_score": risk_score,
                "risk_level": risk_level
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/events", response_model=EventResponse)
def log_event(event: EventCreate):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = time.time()
    cursor.execute(
        "INSERT INTO events (user_id, timestamp, class_name, primary_conf, verification_conf, risk_score, risk_level) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (event.user_id, timestamp, event.class_name, event.primary_conf, event.verification_conf, event.risk_score, event.risk_level)
    )
    event_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {
        "id": event_id,
        "user_id": event.user_id,
        "timestamp": timestamp,
        "class_name": event.class_name,
        "primary_conf": event.primary_conf,
        "verification_conf": event.verification_conf,
        "risk_score": event.risk_score,
        "risk_level": event.risk_level
    }

@app.get("/events/{user_id}", response_model=List[EventResponse])
def get_event_history(user_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM events WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.post("/contacts", response_model=ContactResponse)
def add_contact(contact: ContactCreate):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO contacts (user_id, name, phone, relation) VALUES (?, ?, ?, ?)",
        (contact.user_id, contact.name, contact.phone, contact.relation)
    )
    contact_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {
        "id": contact_id,
        "user_id": contact.user_id,
        "name": contact.name,
        "phone": contact.phone,
        "relation": contact.relation
    }

@app.get("/contacts/{user_id}", response_model=List[ContactResponse])
def get_contacts(user_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM contacts WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.delete("/contacts/{contact_id}")
def delete_contact(contact_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
    conn.commit()
    conn.close()
    return {"status": "success", "message": f"Contact {contact_id} deleted"}

@app.get("/nearby")
def get_nearby_places(
    lat: float = Query(..., description="Latitude"),
    lng: float = Query(..., description="Longitude"),
    place_type: str = Query(..., alias="type", description="Type of service: hospital, police, fire")
):
    osm_node_type = "amenity"
    osm_val = ""
    if place_type == "hospital":
        osm_val = "hospital"
    elif place_type == "police":
        osm_val = "police"
    elif place_type == "fire":
        osm_val = "fire_station"
    else:
        raise HTTPException(status_code=400, detail="Invalid place type.")
        
    overpass_query = f"""
    [out:json];
    node["{osm_node_type}"="{osm_val}"](around:5000, {lat}, {lng});
    out body;
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    try:
        response = requests.post(overpass_url, data={"data": overpass_query}, timeout=10)
        response.raise_for_status()
        data = response.json()
        places = []
        for element in data.get("elements", []):
            tags = element.get("tags", {})
            places.append({
                "name": tags.get("name", f"Unnamed {place_type.capitalize()}"),
                "latitude": element.get("lat"),
                "longitude": element.get("lon"),
                "address": tags.get("addr:street", "Street address unavailable")
            })
        return {"status": "success", "results": places[:10]}
    except Exception as e:
        return {
            "status": "fallback",
            "simulated": True,
            "message": "OSM Overpass API failure. Using local mock coordinates.",
            "results": [
                {
                    "name": f"City Emergency {place_type.capitalize()}",
                    "latitude": lat + 0.005,
                    "longitude": lng - 0.003,
                    "address": "123 Civic Center Way"
                },
                {
                    "name": f"Central District {place_type.capitalize()}",
                    "latitude": lat - 0.002,
                    "longitude": lng + 0.006,
                    "address": "456 Safety Blvd"
                }
            ]
        }

@app.post("/demo/nearby-corroboration")
def get_demo_corroboration(lat: float, lng: float, class_name: str):
    return {
        "simulated": True,
        "active_danger_zone": True if class_name in ["gunshot", "explosion"] else False,
        "corroborated_reports_count": 3,
        "time_window_minutes": 5,
        "alert_corroborated": True
    }

# Mount Data static directory
data_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "data"))
os.makedirs(data_dir_path, exist_ok=True)
app.mount("/data", StaticFiles(directory=data_dir_path), name="data")

# Mount Web Emulator static assets
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
