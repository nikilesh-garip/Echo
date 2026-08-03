import os
import sys
import time
import sqlite3
import requests
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import contextmanager

# Add parent directory to path so we can import model modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model")))

from two_pass_detector import TwoPassDetector
from risk_scorer import RiskScorer

DB_PATH = "echo_backend.db"
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "checkpoints", "best_model.pth"))

@contextmanager
def get_db():
    """Context manager to prevent SQLite connection leaks under runtime exceptions."""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
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
                risk_level TEXT NOT NULL,
                latitude REAL DEFAULT 0.0,
                longitude REAL DEFAULT 0.0
            )
        """)
        # Migration: Add latitude and longitude to events if they do not exist
        try:
            cursor.execute("ALTER TABLE events ADD COLUMN latitude REAL DEFAULT 0.0")
        except sqlite3.OperationalError:
            pass # already exists
        try:
            cursor.execute("ALTER TABLE events ADD COLUMN longitude REAL DEFAULT 0.0")
        except sqlite3.OperationalError:
            pass # already exists
            
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

init_db()

app = FastAPI(title="Echo Smart Emergency System API", version="1.0")

# Initialize Detector and Scorer (Fail fast on startup if checkpoint is missing or corrupt)
if not os.path.exists(MODEL_PATH):
    print(f"CRITICAL: Model checkpoint missing at: {MODEL_PATH}")
    sys.exit(1)

try:
    detector = TwoPassDetector(MODEL_PATH)
    print("Successfully loaded CRNN model for inference!")
except Exception as e:
    print(f"CRITICAL ERROR loading model: {e}")
    sys.exit(1)

scorer = RiskScorer()

# Pydantic Schemas
class EventCreate(BaseModel):
    user_id: str
    class_name: str
    primary_conf: float
    verification_conf: float
    risk_score: int
    risk_level: str
    latitude: Optional[float] = 0.0
    longitude: Optional[float] = 0.0

class EventResponse(BaseModel):
    id: int
    user_id: str
    timestamp: float
    class_name: str
    primary_conf: float
    verification_conf: float
    risk_score: int
    risk_level: str
    latitude: Optional[float] = 0.0
    longitude: Optional[float] = 0.0

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
        
    temp_path = None
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
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process uploaded audio: {str(e)}")
    finally:
        # Prevent temporary file leaks by ensuring cleanup on failure
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
                
    try:
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
    with get_db() as conn:
        cursor = conn.cursor()
        timestamp = time.time()
        cursor.execute(
            "INSERT INTO events (user_id, timestamp, class_name, primary_conf, verification_conf, risk_score, risk_level, latitude, longitude) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (event.user_id, timestamp, event.class_name, event.primary_conf, event.verification_conf, event.risk_score, event.risk_level, event.latitude, event.longitude)
        )
        event_id = cursor.lastrowid
        conn.commit()
        
    return {
        "id": event_id,
        "user_id": event.user_id,
        "timestamp": timestamp,
        "class_name": event.class_name,
        "primary_conf": event.primary_conf,
        "verification_conf": event.verification_conf,
        "risk_score": event.risk_score,
        "risk_level": event.risk_level,
        "latitude": event.latitude,
        "longitude": event.longitude
    }

@app.get("/events/{user_id}", response_model=List[EventResponse])
def get_event_history(user_id: str):
    with get_db() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM events WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
        rows = cursor.fetchall()
        
    return [dict(row) for row in rows]

@app.post("/contacts", response_model=ContactResponse)
def add_contact(contact: ContactCreate):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO contacts (user_id, name, phone, relation) VALUES (?, ?, ?, ?)",
            (contact.user_id, contact.name, contact.phone, contact.relation)
        )
        contact_id = cursor.lastrowid
        conn.commit()
        
    return {
        "id": contact_id,
        "user_id": contact.user_id,
        "name": contact.name,
        "phone": contact.phone,
        "relation": contact.relation
    }

@app.get("/contacts/{user_id}", response_model=List[ContactResponse])
def get_contacts(user_id: str):
    with get_db() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM contacts WHERE user_id = ?", (user_id,))
        rows = cursor.fetchall()
        
    return [dict(row) for row in rows]

@app.delete("/contacts/{contact_id}")
def delete_contact(contact_id: int):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
        conn.commit()
        
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
    nwr["{osm_node_type}"="{osm_val}"](around:5000, {lat}, {lng});
    out center;
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    try:
        response = requests.post(overpass_url, data={"data": overpass_query}, timeout=10)
        response.raise_for_status()
        data = response.json()
        places = []
        for element in data.get("elements", []):
            tags = element.get("tags", {})
            el_lat = element.get("lat") or (element.get("center", {}).get("lat") if element.get("center") else None)
            el_lon = element.get("lon") or (element.get("center", {}).get("lon") if element.get("center") else None)
            if el_lat is not None and el_lon is not None:
                places.append({
                    "name": tags.get("name", tags.get("official_name", f"Unnamed {place_type.capitalize()}")),
                    "latitude": el_lat,
                    "longitude": el_lon,
                    "address": tags.get("addr:street", tags.get("addr:place", "Street address unavailable"))
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
