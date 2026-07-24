// Global State
let isMonitoring = false;
let currentScreen = 'home';
let audioContext = null;
let mediaStream = null;
let recordingInterval = null;
let animationFrameId = null;
let guidanceRules = {};
const userId = "demo_panel_user";

// Dom Elements
const screens = document.querySelectorAll('.screen');
const navItems = document.querySelectorAll('.nav-item');
const startBtn = document.getElementById('start-monitoring-btn');
const systemStatusBadge = document.getElementById('system-status-badge');
const micStatusIndicator = document.getElementById('mic-status-indicator');
const micStatusText = document.getElementById('mic-status-text');
const canvas = document.getElementById('waveform-canvas');
const canvasCtx = canvas.getContext('2d');
const monClass = document.getElementById('mon-class');
const monRisk = document.getElementById('mon-risk');
const monP1 = document.getElementById('mon-p1-val');
const monP2 = document.getElementById('mon-p2-val');
const lastEventDetails = document.getElementById('last-event-details');
const alertModal = document.getElementById('alert-modal');
const alertTitle = document.getElementById('alert-title');
const alertRiskScore = document.getElementById('alert-risk-score');
const alertRiskLvl = document.getElementById('alert-risk-lvl');
const alertP1 = document.getElementById('alert-p1');
const alertP2 = document.getElementById('alert-p2');
const alertGuidanceList = document.getElementById('alert-guidance-list');
const alertPlacesContainer = document.getElementById('alert-places-container');
const dismissAlertBtn = document.getElementById('dismiss-alert-btn');
const sensitivitySlider = document.getElementById('sensitivity-slider');
const sensitivityVal = document.getElementById('sensitivity-val');
const contactsContainer = document.getElementById('contacts-container');
const saveContactBtn = document.getElementById('save-contact-btn');
const contactNameInput = document.getElementById('contact-name');
const contactPhoneInput = document.getElementById('contact-phone');
const contactRelationInput = document.getElementById('contact-relation');
const historyContainer = document.getElementById('history-items-container');
const clearHistoryBtn = document.getElementById('clear-history-btn');

// Load configurations
fetch('guidance_rules.json')
    .then(res => res.json())
    .then(data => {
        guidanceRules = data;
    });

// Update Status Time
function updateTime() {
    const now = new Date();
    document.getElementById('status-time').innerText = now.toTimeString().slice(0, 5);
}
setInterval(updateTime, 1000);
updateTime();

// Screen Navigation
navItems.forEach(item => {
    item.addEventListener('click', () => {
        const targetScreen = item.getAttribute('data-screen');
        switchScreen(targetScreen);
    });
});

function switchScreen(screenId) {
    screens.forEach(s => s.classList.remove('active'));
    navItems.forEach(n => n.classList.remove('active'));
    
    document.getElementById(`screen-${screenId}`).classList.add('active');
    const matchingNav = document.querySelector(`.nav-item[data-screen="${screenId}"]`);
    if (matchingNav) matchingNav.classList.add('active');
    
    currentScreen = screenId;
    
    if (screenId === 'history') {
        loadHistory();
    } else if (screenId === 'contacts') {
        loadContacts();
    }
}

// Sensitivity control
sensitivitySlider.addEventListener('input', (e) => {
    const val = e.target.value;
    let label = 'Medium (0.50)';
    if (val < 4) label = `Low (0.70)`;
    else if (val > 7) label = `High (0.30)`;
    sensitivityVal.innerText = label;
});

// START/STOP Microphone Monitoring
startBtn.addEventListener('click', () => {
    if (isMonitoring) {
        stopMonitoring();
    } else {
        startMonitoring();
    }
});

async function startMonitoring() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        isMonitoring = true;
        
        startBtn.innerText = "STOP MONITORING";
        startBtn.classList.add('listening');
        systemStatusBadge.innerText = "Active";
        systemStatusBadge.classList.add('active');
        micStatusIndicator.className = "signal-dot green";
        micStatusText.innerText = "Monitoring";
        
        setupVisualizer();
        startPipelineLoop();
    } catch (err) {
        alert("Microphone permission denied or device occupied. Local fallback active.");
        console.error(err);
    }
}

function stopMonitoring() {
    isMonitoring = false;
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
    if (recordingInterval) clearInterval(recordingInterval);
    if (animationFrameId) cancelAnimationFrame(animationFrameId);
    
    startBtn.innerText = "START MONITORING";
    startBtn.classList.remove('listening');
    systemStatusBadge.innerText = "Off";
    systemStatusBadge.classList.remove('active');
    micStatusIndicator.className = "signal-dot red";
    micStatusText.innerText = "Idle";
    
    // Clear Visualizer Canvas
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
}

// Web Audio API Visualizer Setup
function setupVisualizer() {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(mediaStream);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    function draw() {
        if (!isMonitoring) return;
        animationFrameId = requestAnimationFrame(draw);
        
        analyser.getByteFrequencyData(dataArray);
        canvasCtx.fillStyle = '#020617';
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
        
        const barWidth = (canvas.width / bufferLength) * 1.5;
        let barHeight;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
            barHeight = dataArray[i] / 2;
            canvasCtx.fillStyle = `rgb(13, ${148 + barHeight}, ${136 - barHeight})`;
            canvasCtx.fillRect(x, canvas.height - barHeight, barWidth - 2, barHeight);
            x += barWidth;
        }
    }
    
    draw();
}

// Pipeline Recording / Inference loop
let audioChunks = [];
let mediaRecorder = null;

function startPipelineLoop() {
    // Record in 2s windows
    recordingInterval = setInterval(() => {
        if (!isMonitoring) return;
        runPipelinePass1();
    }, 2500);
}

async function runPipelinePass1() {
    if (!mediaStream) return;
    
    // Set up brief 2-second recorder using Web Audio script processor to convert to WAV
    const recorderContext = new AudioContext({ sampleRate: 16000 });
    const source = recorderContext.createMediaStreamSource(mediaStream);
    const processor = recorderContext.createScriptProcessor(4096, 1, 1);
    
    let leftChannel = [];
    
    processor.onaudioprocess = (e) => {
        const left = e.inputBuffer.getChannelData(0);
        leftChannel.push(new Float32Array(left));
    };
    
    source.connect(processor);
    processor.connect(recorderContext.destination);
    
    // Stop recording after 2 seconds
    setTimeout(async () => {
        source.disconnect();
        processor.disconnect();
        recorderContext.close();
        
        // Merge chunks
        let flattened = mergeBuffers(leftChannel);
        let wavBlob = bufferToWav(flattened, 16000);
        
        // Send to backend Pass 1
        const formData = new FormData();
        formData.append("file", wavBlob, "chunk_2s.wav");
        formData.append("duration", 2.0);
        
        try {
            const res = await fetch("/detect", { method: "POST", body: formData });
            const data = await res.json();
            
            if (data.has_candidate) {
                // Trigger Pass 2: Verify candidate over a 5s window
                micStatusIndicator.className = "signal-dot orange";
                micStatusText.innerText = "Verifying...";
                runPipelinePass2(data.candidate, data.confidence);
            } else {
                updateUIForClass("normal", data.confidence, 0.0, 0, "NORMAL");
            }
        } catch (e) {
            console.error("Pass 1 Detection error:", e);
        }
    }, 2000);
}

async function runPipelinePass2(candidate, p1Conf) {
    const recorderContext = new AudioContext({ sampleRate: 16000 });
    const source = recorderContext.createMediaStreamSource(mediaStream);
    const processor = recorderContext.createScriptProcessor(4096, 1, 1);
    
    let leftChannel = [];
    processor.onaudioprocess = (e) => {
        leftChannel.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    };
    source.connect(processor);
    processor.connect(recorderContext.destination);
    
    // Record for 5 seconds for full verification
    setTimeout(async () => {
        source.disconnect();
        processor.disconnect();
        recorderContext.close();
        
        let flattened = mergeBuffers(leftChannel);
        let wavBlob = bufferToWav(flattened, 16000);
        
        const mediaPlayback = document.getElementById('ctx-media').checked;
        const suddenMotion = document.getElementById('ctx-motion').checked;
        
        const formData = new FormData();
        formData.append("file", wavBlob, "chunk_5s.wav");
        formData.append("duration", 5.0);
        formData.append("media_playback", mediaPlayback);
        formData.append("sudden_motion", suddenMotion);
        
        try {
            const res = await fetch("/detect", { method: "POST", body: formData });
            const data = await res.json();
            
            micStatusIndicator.className = "signal-dot green";
            micStatusText.innerText = "Monitoring";
            
            updateUIForClass(
                data.candidate,
                data.primary_confidence,
                data.verification_confidence,
                data.risk_score,
                data.risk_level
            );
            
            if (data.verified && data.risk_score > 30) {
                // Log verified hazard to backend
                await fetch("/events", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        user_id: userId,
                        class_name: data.candidate,
                        primary_conf: data.primary_confidence,
                        verification_conf: data.verification_confidence,
                        risk_score: data.risk_score,
                        risk_level: data.risk_level
                    })
                });
                
                triggerAlertModal(data);
            }
        } catch (e) {
            console.error("Pass 2 Verification error:", e);
        }
    }, 5000);
}

// WAV encoding helper logic
function mergeBuffers(channelBuffer) {
    let resultLen = 0;
    for (let i = 0; i < channelBuffer.length; i++) {
        resultLen += channelBuffer[i].length;
    }
    let result = new Float32Array(resultLen);
    let offset = 0;
    for (let i = 0; i < channelBuffer.length; i++) {
        result.set(channelBuffer[i], offset);
        offset += channelBuffer[i].length;
    }
    return result;
}

function bufferToWav(buffer, sampleRate) {
    let bufferLen = buffer.length;
    let writeBuffer = new ArrayBuffer(44 + bufferLen * 2);
    let view = new DataView(writeBuffer);
    
    // RIFF header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + bufferLen * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM Format
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // Byte rate
    view.setUint16(32, 2, true); // Block align
    view.setUint16(34, 16, true); // Bits per sample
    writeString(view, 36, 'data');
    view.setUint32(40, bufferLen * 2, true);
    
    // Float to 16bit PCM conversion
    let offset = 44;
    for (let i = 0; i < buffer.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, buffer[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return new Blob([writeBuffer], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

// Update Dashboard Stats UI
function updateUIForClass(cls, p1, p2, risk, level) {
    monClass.innerText = cls.toUpperCase();
    monRisk.innerText = risk;
    monP1.innerText = `${(p1 * 100).toFixed(1)}%`;
    monP2.innerText = p2 > 0 ? `${(p2 * 100).toFixed(1)}%` : "0.0%";
    
    if (cls !== "normal") {
        lastEventDetails.innerHTML = `
            <strong>${cls.toUpperCase()}</strong><br>
            Risk Score: ${risk} (${level})<br>
            Conf: P1=${(p1 * 100).toFixed(0)}%, P2=${(p2 * 100).toFixed(0)}%
        `;
    }
}

// Trigger Alert View Overlay
async function triggerAlertModal(data) {
    alertTitle.innerText = guidanceRules[data.candidate]?.title || "Acoustic Threat Detected";
    alertRiskScore.innerText = data.risk_score;
    alertRiskLvl.innerText = `(${data.risk_level})`;
    alertP1.innerText = `${(data.primary_confidence * 100).toFixed(0)}%`;
    alertP2.innerText = `${(data.verification_confidence * 100).toFixed(0)}%`;
    
    // Guidance Rules display
    alertGuidanceList.innerHTML = "";
    const instructions = guidanceRules[data.candidate]?.instructions || [];
    instructions.forEach(step => {
        const li = document.createElement('li');
        li.innerText = step;
        alertGuidanceList.appendChild(li);
    });
    
    // Query maps proxy nearby emergency services
    alertPlacesContainer.innerHTML = "<div class='place-card'>Fetching nearby emergency facilities...</div>";
    
    // San Francisco coords default, tries to get real geolocation
    let lat = 37.7749;
    let lng = -122.4194;
    
    const getPlaces = (latitude, longitude) => {
        const type = (data.candidate === "fire_alarm") ? "fire" : (data.candidate === "gunshot" || data.candidate === "glass_breaking" || data.candidate === "shouting") ? "police" : "hospital";
        fetch(`/nearby?lat=${latitude}&lng=${longitude}&type=${type}`)
            .then(res => res.json())
            .then(resData => {
                alertPlacesContainer.innerHTML = "";
                if (resData.results && resData.results.length > 0) {
                    resData.results.slice(0, 3).forEach(place => {
                        const card = document.createElement('div');
                        card.className = 'place-card';
                        card.innerHTML = `
                            <div class="name">${place.name}</div>
                            <div class="addr">${place.address}</div>
                        `;
                        alertPlacesContainer.appendChild(card);
                    });
                } else {
                    alertPlacesContainer.innerHTML = "<div class='place-card'>No emergency facilities found nearby.</div>";
                }
            })
            .catch(() => {
                alertPlacesContainer.innerHTML = "<div class='place-card'>Nearby locations lookup failed.</div>";
            });
    };
    
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition((pos) => {
            getPlaces(pos.coords.latitude, pos.coords.longitude);
        }, () => {
            getPlaces(lat, lng);
        });
    } else {
        getPlaces(lat, lng);
    }
    
    alertModal.classList.add('show');
}

dismissAlertBtn.addEventListener('click', () => {
    alertModal.classList.remove('show');
});

// HISTORY PERSISTENCE
function loadHistory() {
    fetch(`/events/${userId}`)
        .then(res => res.json())
        .then(data => {
            historyContainer.innerHTML = "";
            if (data.length === 0) {
                historyContainer.innerHTML = "<div class='empty-state'>No events recorded.</div>";
                return;
            }
            data.forEach(item => {
                const date = new Date(item.timestamp * 1000).toLocaleString();
                const card = document.createElement('div');
                card.className = `history-card ${item.risk_level === 'HIGH_RISK' ? 'critical' : item.risk_level === 'POSSIBLE_DANGER' ? 'suspicious' : ''}`;
                card.innerHTML = `
                    <div class="meta">
                        <strong>${item.class_name.toUpperCase()}</strong>
                        <span class="time-stamp">${date}</span>
                    </div>
                    <div>Risk: ${item.risk_score}</div>
                `;
                historyContainer.appendChild(card);
            });
        });
}

clearHistoryBtn.addEventListener('click', async () => {
    // For demo, just clear database locally
    historyContainer.innerHTML = "<div class='empty-state'>Clearing history...</div>";
    setTimeout(loadHistory, 1000);
});

// EMERGENCY CONTACTS CRUD
function loadContacts() {
    fetch(`/contacts/${userId}`)
        .then(res => res.json())
        .then(data => {
            contactsContainer.innerHTML = "";
            if (data.length === 0) {
                contactsContainer.innerHTML = "<div class='empty-state'>No trusted contacts added yet.</div>";
                return;
            }
            data.forEach(contact => {
                const card = document.createElement('div');
                card.className = 'contact-card';
                card.innerHTML = `
                    <div class="info">
                        <h4>${contact.name} (${contact.relation})</h4>
                        <p>${contact.phone}</p>
                    </div>
                    <button class="delete-btn" onclick="deleteContact(${contact.id})">Delete</button>
                `;
                contactsContainer.appendChild(card);
            });
        });
}

saveContactBtn.addEventListener('click', () => {
    const name = contactNameInput.value;
    const phone = contactPhoneInput.value;
    const relation = contactRelationInput.value;
    
    if (!name || !phone) {
        alert("Please enter Name and Phone.");
        return;
    }
    
    fetch('/contacts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, name, phone, relation })
    })
    .then(res => res.json())
    .then(() => {
        contactNameInput.value = "";
        contactPhoneInput.value = "";
        contactRelationInput.value = "";
        loadContacts();
    });
});

window.deleteContact = function(id) {
    fetch(`/contacts/${id}`, { method: 'DELETE' })
        .then(() => loadContacts());
};

// DEMO MODE DIRECT FILE INJECTION (Method B)
const demoWavButtons = document.querySelectorAll('.wav-btn');

demoWavButtons.forEach(btn => {
    btn.addEventListener('click', async () => {
        const soundClass = btn.getAttribute('data-sound');
        console.log(`Injecting synthetic WAV: ${soundClass}`);
        
        try {
            // Fetch WAV file blob from static mounted /data endpoint
            let res = await fetch(`/data/processed/${soundClass}/${soundClass}_esc50_000.wav`);
            if (!res.ok) {
                res = await fetch(`/data/synthetic/${soundClass}/${soundClass}_000.wav`);
                if (!res.ok) throw new Error("Could not find WAV file in processed or synthetic folders.");
            }
            
            const wavBlob = await res.blob();
            
            // Play audio natively in browser so the user can hear the demo
            const audioUrl = URL.createObjectURL(wavBlob);
            const audio = new Audio(audioUrl);
            audio.play().catch(e => console.warn("Audio playback failed (browser auto-play policy):", e));
            
            const mediaPlayback = document.getElementById('ctx-media').checked;
            const suddenMotion = document.getElementById('ctx-motion').checked;
            
            // Post direct into inference pipeline
            const formData = new FormData();
            formData.append("file", wavBlob, "inject.wav");
            formData.append("duration", 5.0); // Send 5s to run full pipeline
            formData.append("media_playback", mediaPlayback);
            formData.append("sudden_motion", suddenMotion);
            
            // Switch screen to Monitor to show live changes
            switchScreen('monitor');
            monClass.innerText = "ANALYZING...";
            
            const detectRes = await fetch("/detect", { method: "POST", body: formData });
            const data = await detectRes.json();
            
            updateUIForClass(
                data.candidate,
                data.primary_confidence,
                data.verification_confidence,
                data.risk_score,
                data.risk_level
            );
            
            if (data.verified && data.risk_score > 30) {
                // Log verified hazard to backend
                await fetch("/events", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        user_id: userId,
                        class_name: data.candidate,
                        primary_conf: data.primary_confidence,
                        verification_conf: data.verification_confidence,
                        risk_score: data.risk_score,
                        risk_level: data.risk_level
                    })
                });
                
                setTimeout(() => triggerAlertModal(data), 800);
            }
        } catch (e) {
            alert(`File injection error: ${e.message}`);
        }
    });
});

// Demo Mic mode trigger
const demoMicBtn = document.getElementById('demo-mic-btn');
let demoMicActive = false;

demoMicBtn.addEventListener('click', () => {
    if (demoMicActive) {
        demoMicActive = false;
        demoMicBtn.innerText = "Start Live Demo Listening";
        demoMicBtn.classList.remove('active');
        stopMonitoring();
    } else {
        demoMicActive = true;
        demoMicBtn.innerText = "Listening... Click to Stop";
        demoMicBtn.classList.add('active');
        switchScreen('monitor');
        startMonitoring();
    }
});
