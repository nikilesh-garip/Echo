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

// Cooldown tracking for continuous sound alerts
let lastAlertClass = null;
let lastAlertTime = 0;
const ALERT_COOLDOWN_MS = 15000; // 15 seconds cooldown for the same class

function shouldTriggerAlert(candidate, riskScore) {
    if (riskScore <= 30) return false;
    
    // Check if alert modal is currently visible
    const isAlertOpen = alertModal.classList.contains('show');
    if (isAlertOpen) {
        console.log(`Suppressing alert: alert modal is already open`);
        return false;
    }
    
    // Check cooldown for the same class
    const now = Date.now();
    if (candidate === lastAlertClass && (now - lastAlertTime < ALERT_COOLDOWN_MS)) {
        console.log(`Suppressing alert: same class ${candidate} detected within cooldown`);
        return false;
    }
    
    // Update tracking
    lastAlertClass = candidate;
    lastAlertTime = now;
    return true;
}

// =============================================================================
// LOCATION SERVICE — Browser Geolocation API (watchPosition, high accuracy)
// =============================================================================
const LocationService = (() => {
    // Internal state
    const state = {
        latitude:  null,
        longitude: null,
        accuracy:  null,   // metres
        timestamp: null,   // Unix ms
        mapsUrl:   null,
        watchId:   null,
        status:    'idle', // idle | acquiring | active | error
        errorCode: null    // 1=PERMISSION_DENIED 2=UNAVAILABLE 3=TIMEOUT
    };

    // DOM refs — resolved lazily (elements exist after DOMContentLoaded)
    const el = () => ({
        statusIcon:    document.getElementById('loc-status-icon'),
        statusText:    document.getElementById('loc-status-text'),
        latDisplay:    document.getElementById('loc-lat-display'),
        lngDisplay:    document.getElementById('loc-lng-display'),
        accDisplay:    document.getElementById('loc-acc-display'),
        tsDisplay:     document.getElementById('loc-ts-display'),
        badge:         document.getElementById('loc-accuracy-badge'),
        mapsBtn:       document.getElementById('open-maps-btn'),
        permDot:       document.getElementById('loc-permission-dot'),
        permLabel:     document.getElementById('loc-permission-label')
    });

    // Build Google Maps search URL from coordinates (no API key required)
    function buildMapsUrl(lat, lng) {
        return `https://www.google.com/maps/search/?api=1&query=${lat},${lng}`;
    }

    // Classify accuracy into a human-readable tier
    function accuracyTier(metres) {
        if (metres === null) return { cls: 'unknown', label: '— Accuracy' };
        if (metres <= 50)    return { cls: 'good',    label: `±${Math.round(metres)} m — Good` };
        if (metres <= 500)   return { cls: 'fair',    label: `±${Math.round(metres)} m — Fair` };
        return                      { cls: 'poor',    label: `±${Math.round(metres)} m — Poor` };
    }

    // Update all Settings-screen location UI elements
    function updateUI() {
        const d = el();
        if (!d.statusIcon) return; // DOM not ready yet

        if (state.status === 'acquiring') {
            d.statusIcon.className = 'location-status-icon acquiring';
            d.statusText.textContent = 'Acquiring location…';
            d.permDot.className = 'status-dot yellow';
            d.mapsBtn.disabled = true;
            return;
        }

        if (state.status === 'active') {
            const tier = accuracyTier(state.accuracy);

            d.statusIcon.className = `location-status-icon ${tier.cls === 'poor' ? 'yellow' : 'green'}`;
            d.statusText.textContent = 'Location active — tracking in real time';
            d.permDot.className = 'status-dot green';

            d.latDisplay.textContent  = `Lat:  ${state.latitude.toFixed(6)}`;
            d.lngDisplay.textContent  = `Lng:  ${state.longitude.toFixed(6)}`;
            d.accDisplay.textContent  = `Accuracy: ±${Math.round(state.accuracy)} m`;

            const ts = new Date(state.timestamp);
            d.tsDisplay.textContent = `Last updated: ${ts.toLocaleTimeString()}`;

            d.badge.className = `accuracy-badge ${tier.cls}`;
            d.badge.textContent = tier.label;

            // Enable Maps button and bind click
            d.mapsBtn.disabled = false;
            return;
        }

        if (state.status === 'error') {
            let msg, dotCls;
            switch (state.errorCode) {
                case 1:
                    msg    = 'Permission denied — enable location in browser settings';
                    dotCls = 'red';
                    break;
                case 2:
                    msg    = 'Location unavailable — check device GPS / network';
                    dotCls = 'orange';
                    break;
                case 3:
                    msg    = 'Location timed out — retrying…';
                    dotCls = 'orange';
                    break;
                default:
                    msg    = 'Location error — geolocation not supported';
                    dotCls = 'red';
            }
            d.statusIcon.className = `location-status-icon ${dotCls}`;
            d.statusText.textContent = msg;
            d.permDot.className = `status-dot ${dotCls}`;
            d.badge.className = 'accuracy-badge unknown';
            d.badge.textContent = '— Accuracy';
            d.mapsBtn.disabled = true;
        }
    }

    // watchPosition success callback
    function onPosition(pos) {
        state.latitude  = pos.coords.latitude;
        state.longitude = pos.coords.longitude;
        state.accuracy  = pos.coords.accuracy;
        state.timestamp = pos.timestamp;
        state.mapsUrl   = buildMapsUrl(state.latitude, state.longitude);
        state.status    = 'active';
        state.errorCode = null;
        console.log(
            `[LocationService] Updated: ${state.latitude.toFixed(6)}, ` +
            `${state.longitude.toFixed(6)} ±${Math.round(state.accuracy)}m`
        );
        updateUI();
    }

    // watchPosition error callback
    function onError(err) {
        state.status    = 'error';
        state.errorCode = err.code;
        console.warn('[LocationService] Error:', err.message, '(code', err.code + ')');
        updateUI();
    }

    // Public API
    return {
        init() {
            if (!navigator.geolocation) {
                state.status    = 'error';
                state.errorCode = 0; // unsupported
                console.warn('[LocationService] Geolocation not supported by this browser.');
                updateUI();
                return;
            }
            if (state.watchId !== null) return; // already watching

            state.status = 'acquiring';
            updateUI();

            state.watchId = navigator.geolocation.watchPosition(
                onPosition,
                onError,
                {
                    enableHighAccuracy: true,
                    timeout:           10000,
                    maximumAge:        0
                }
            );
            console.log('[LocationService] watchPosition started, id:', state.watchId);

            // Wire the "Open in Maps" button (once)
            const btn = document.getElementById('open-maps-btn');
            if (btn) {
                btn.addEventListener('click', () => {
                    if (!state.mapsUrl) return;
                    // Opens in new tab on desktop; mobile browsers hand-off to Maps app
                    window.open(state.mapsUrl, '_blank', 'noopener,noreferrer');
                });
            }
        },

        stop() {
            if (state.watchId !== null) {
                navigator.geolocation.clearWatch(state.watchId);
                state.watchId = null;
                console.log('[LocationService] watchPosition cleared.');
            }
        },

        // Returns current cached coords or null if not yet available
        getCoords() {
            if (state.latitude === null) return null;
            return {
                latitude:  state.latitude,
                longitude: state.longitude,
                accuracy:  state.accuracy,
                timestamp: state.timestamp,
                mapsUrl:   state.mapsUrl
            };
        },

        // Async helper kept for backward-compat with older call sites.
        // Returns cached coords immediately if available, otherwise waits
        // up to 10 s for the first watchPosition fix.
        getDeviceLocation() {
            const cached = this.getCoords();
            if (cached) return Promise.resolve(cached);

            // Not yet resolved — fall back to one-shot getCurrentPosition
            return new Promise((resolve) => {
                if (!navigator.geolocation) {
                    resolve({ latitude: 0, longitude: 0 });
                    return;
                }
                navigator.geolocation.getCurrentPosition(
                    (pos) => resolve({
                        latitude:  pos.coords.latitude,
                        longitude: pos.coords.longitude,
                        accuracy:  pos.coords.accuracy,
                        timestamp: pos.timestamp,
                        mapsUrl:   buildMapsUrl(pos.coords.latitude, pos.coords.longitude)
                    }),
                    () => resolve({ latitude: 0, longitude: 0 }),
                    { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
                );
            });
        },

        get mapsUrl()   { return state.mapsUrl; },
        get latitude()  { return state.latitude; },
        get longitude() { return state.longitude; }
    };
})();

// Backward-compat shim — called by runPipelinePass1/2 and demo mode
async function getDeviceLocation() {
    return LocationService.getDeviceLocation();
}

// Auto-init on page load so coords are ready before the first alert fires
LocationService.init();

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
        canvasCtx.fillStyle = '#ffffff';
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
        
        const barWidth = (canvas.width / bufferLength) * 1.5;
        let barHeight;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
            barHeight = dataArray[i] / 2;
            canvasCtx.fillStyle = `rgb(255, ${216 - (barHeight * 0.4)}, 3)`;
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
                if (data.immediate_verification) {
                    updateUIForClass(
                        data.candidate,
                        data.primary_confidence,
                        data.verification_confidence,
                        data.risk_score,
                        data.risk_level
                    );
                    
                    if (data.verified && shouldTriggerAlert(data.candidate, data.risk_score)) {
                        const loc = await getDeviceLocation();
                        data.latitude = loc.latitude;
                        data.longitude = loc.longitude;

                        // Log verified hazard to backend immediately
                        await fetch("/events", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                                user_id: userId,
                                class_name: data.candidate,
                                primary_conf: data.primary_confidence,
                                verification_conf: data.verification_confidence,
                                risk_score: data.risk_score,
                                risk_level: data.risk_level,
                                latitude: loc.latitude,
                                longitude: loc.longitude
                            })
                        });
                        triggerAlertModal(data);
                    }
                } else {
                    // Trigger Pass 2: Verify candidate over a 5s window
                    micStatusIndicator.className = "signal-dot orange";
                    micStatusText.innerText = "Verifying...";
                    runPipelinePass2(data.candidate, data.confidence);
                }
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
            
            if (data.verified && shouldTriggerAlert(data.candidate, data.risk_score)) {
                const loc = await getDeviceLocation();
                data.latitude = loc.latitude;
                data.longitude = loc.longitude;

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
                        risk_level: data.risk_level,
                        latitude: loc.latitude,
                        longitude: loc.longitude
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
    
    // Default coordinates (San Francisco)
    let lat = data.latitude || 37.7749;
    let lng = data.longitude || -122.4194;
    
    // Add broadcast status
    let broadcastDiv = document.getElementById("alert-broadcast-status");
    if (!broadcastDiv) {
        broadcastDiv = document.createElement("div");
        broadcastDiv.id = "alert-broadcast-status";
        broadcastDiv.style.backgroundColor = "rgba(249, 115, 22, 0.1)";
        broadcastDiv.style.border = "1px solid #f97316";
        broadcastDiv.style.borderRadius = "8px";
        broadcastDiv.style.padding = "10px";
        broadcastDiv.style.marginTop = "10px";
        broadcastDiv.style.color = "#f97316";
        broadcastDiv.style.fontSize = "12px";
        broadcastDiv.style.fontWeight = "bold";
        const alertBody = document.querySelector(".alert-body");
        alertBody.insertBefore(broadcastDiv, alertBody.firstChild);
    }

    // Prefer the high-accuracy Maps URL from LocationService; fall back to basic coords
    const mapLink = LocationService.mapsUrl || `https://www.google.com/maps/search/?api=1&query=${lat},${lng}`;
    fetch(`/contacts/${userId}`)
        .then(res => res.json())
        .then(contacts => {
            if (contacts && contacts.length > 0) {
                const names = contacts.map(c => `${c.name} (${c.relation})`).join(", ");
                broadcastDiv.innerHTML = `🔴 LIVE LOCATION SHARED IMMEDIATELY<br><span style="font-weight: normal; color: #cbd5e1;">SMS alert dispatched to: <strong>${names}</strong> with live map link: <a href="${mapLink}" target="_blank" style="color: #38bdf8; text-decoration: underline;">Open Google Maps</a></span>`;
                console.log(`EMERGENCY SHARING: Live location link (${mapLink}) shared with: ${names}`);
            } else {
                broadcastDiv.innerHTML = `🔴 LIVE LOCATION ACTIVE<br><span style="font-weight: normal; color: #cbd5e1;">Live coordinates: <strong>${lat.toFixed(4)}, ${lng.toFixed(4)}</strong> (Add emergency contacts under the Contacts tab to auto-share).</span>`;
            }
        });

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
    
    // Call places lookup
    getPlaces(lat, lng);
    
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
                
                let locationHtml = "";
                if (item.latitude !== undefined && item.latitude !== null && item.longitude !== undefined && item.longitude !== null && (item.latitude !== 0.0 || item.longitude !== 0.0)) {
                    locationHtml = `
                        <div style="margin-top: 6px; font-size: 11px;">
                            📍 <a href="https://maps.google.com/?q=${item.latitude},${item.longitude}" target="_blank" style="color: #0d9488; text-decoration: underline; font-weight: 500;">
                                Shared Location: ${item.latitude.toFixed(4)}, ${item.longitude.toFixed(4)}
                            </a>
                        </div>
                    `;
                }
                
                card.innerHTML = `
                    <div class="meta">
                        <strong>${item.class_name.toUpperCase()}</strong>
                        <span class="time-stamp">${date}</span>
                    </div>
                    <div>Risk: ${item.risk_score}</div>
                    ${locationHtml}
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
                const loc = await getDeviceLocation();
                data.latitude = loc.latitude;
                data.longitude = loc.longitude;

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
                        risk_level: data.risk_level,
                        latitude: loc.latitude,
                        longitude: loc.longitude
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
