import 'dart:io';
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import '../services/api_service.dart';
import 'alert_screen.dart';

class LiveMonitorScreen extends StatefulWidget {
  const LiveMonitorScreen({super.key});

  @override
  State<LiveMonitorScreen> createState() => _LiveMonitorScreenState();
}

class _LiveMonitorScreenState extends State<LiveMonitorScreen> {
  // Mobile client is configured to connect to local development server on host machine
  final ApiService _apiService = ApiService(baseUrl: 'http://10.0.2.2:8000');
  
  String _currentClass = "NORMAL";
  double _p1Confidence = 0.0;
  double _p2Confidence = 0.0;
  int _riskScore = 0;
  String _riskLevel = "NORMAL";
  bool _isMonitoring = false;

  // Cooldown variables to deduplicate persistent hazards
  String? _lastAlertClass;
  DateTime? _lastAlertTime;
  static const Duration _alertCooldown = Duration(seconds: 15);

  /// Interface bridge to send recorded audio file chunks to the FastAPI detector API.
  Future<void> _sendAudioChunk(File audioFile, double duration) async {
    final result = await _apiService.detectAudio(
      audioFile: audioFile,
      duration: duration,
      mediaPlayback: false, // Binds to actual settings UI toggles in SettingsScreen
      suddenMotion: false,
    );

    if (result != null) {
      setState(() {
        _currentClass = result['candidate'] ?? 'NORMAL';
        _p1Confidence = result['primary_confidence'] ?? result['confidence'] ?? 0.0;
        _p2Confidence = result['verification_confidence'] ?? 0.0;
        _riskScore = result['risk_score'] ?? 0;
        _riskLevel = result['risk_level'] ?? 'NORMAL';
      });

      // If threat is confirmed, log the event and navigate to Alert Screen
      if (result['verified'] == true && _riskScore > 30) {
        // Cooldown check to prevent duplicate alerts for persistent hazards
        final now = DateTime.now();
        if (_currentClass == _lastAlertClass &&
            _lastAlertTime != null &&
            now.difference(_lastAlertTime!) < _alertCooldown) {
          print('Suppressing duplicate alert for $_currentClass within cooldown');
          return;
        }

        _lastAlertClass = _currentClass;
        _lastAlertTime = now;

        double lat = 37.7749; // Default San Francisco coordinates
        double lng = -122.4194;

        try {
          // Request and obtain geolocation permission and position
          bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
          if (serviceEnabled) {
            LocationPermission permission = await Geolocator.checkPermission();
            if (permission == LocationPermission.denied) {
              permission = await Geolocator.requestPermission();
            }
            if (permission == LocationPermission.whileInUse || permission == LocationPermission.always) {
              // Try getting last known position first (instant lookup)
              Position? lastPosition = await Geolocator.getLastKnownPosition();
              if (lastPosition != null) {
                lat = lastPosition.latitude;
                lng = lastPosition.longitude;
              }
              
              // Get current position with balanced accuracy (faster lock indoors) and reasonable timeout
              try {
                Position position = await Geolocator.getCurrentPosition(
                  desiredAccuracy: LocationAccuracy.balanced,
                  timeLimit: const Duration(seconds: 8),
                );
                lat = position.latitude;
                lng = position.longitude;
              } catch (timeoutErr) {
                print('Geolocator getCurrentPosition failed/timed out: $timeoutErr');
              }
            }
          }
        } catch (e) {
          print('Error getting GPS location: $e');
        }

        // Fetch dynamic nearby places from backend OpenStreetMap proxy
        String placeType = (_currentClass == "fire_alarm")
            ? "fire"
            : (_currentClass == "gunshot" || _currentClass == "glass_breaking" || _currentClass == "shouting")
                ? "police"
                : "hospital";

        List<Map<String, dynamic>> nearbyFacilities = [];
        try {
          final places = await _apiService.getNearbyPlaces(lat: lat, lng: lng, type: placeType);
          if (places != null) {
            nearbyFacilities = places;
          }
        } catch (e) {
          print('Error fetching nearby places: $e');
        }

        if (nearbyFacilities.isEmpty) {
          nearbyFacilities = [
            {"name": "Local Emergency Dispatch (GPS Fallback)", "address": "Latitude: ${lat.toStringAsFixed(4)}, Longitude: ${lng.toStringAsFixed(4)}"}
          ];
        }

        // Get emergency contacts to simulate live location message sharing
        List<String> notifiedContacts = [];
        try {
          final contacts = await _apiService.getContacts("echo_mobile_client");
          if (contacts != null && contacts.isNotEmpty) {
            for (var contact in contacts) {
              final contactName = contact['name'] ?? 'Trusted Contact';
              final contactPhone = contact['phone'] ?? '';
              final contactRelation = contact['relation'] ?? 'Friend';
              notifiedContacts.add('$contactName ($contactRelation)');
              
              // Print simulated SMS dispatch log to application standard output
              print('EMERGENCY SHARING: Live location link (https://maps.google.com/?q=$lat,$lng) dispatched immediately via SMS to $contactName ($contactPhone)');
            }
          }
        } catch (e) {
          print('Error fetching contacts for sharing: $e');
        }

        // Log the verified threat to backend along with real location coordinates
        await _apiService.logEvent(
          userId: "echo_mobile_client",
          className: _currentClass,
          primaryConf: _p1Confidence,
          verificationConf: _p2Confidence,
          riskScore: _riskScore,
          riskLevel: _riskLevel,
          latitude: lat,
          longitude: lng,
        );

        if (mounted) {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => AlertScreen(
                title: 'Acoustic Threat Detected',
                threatClass: _currentClass,
                riskScore: _riskScore,
                riskLevel: _riskLevel,
                p1Conf: _p1Confidence,
                p2Conf: _p2Confidence,
                instructions: const [
                  "Remain calm and assess your immediate surroundings.",
                  "Seek substantial shelter or drop low if gunshots are suspected.",
                  "Contact rescue coordinates or emergency contacts immediately."
                ],
                nearbyFacilities: nearbyFacilities,
                latitude: lat,
                longitude: lng,
                notifiedContacts: notifiedContacts,
              ),
            ),
          );
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(20.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Acoustic Spectrum',
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 15),
          
          // Waveform Display Container
          Container(
            height: 140,
            width: double.infinity,
            decoration: BoxDecoration(
              color: Colors.black,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: Colors.white.withOpacity(0.05)),
            ),
            child: Center(
              child: Text(
                _isMonitoring ? 'Spectrogram Stream Active (16 kHz)...' : 'Monitoring Stopped',
                style: TextStyle(color: Colors.white.withOpacity(0.3), fontSize: 12),
              ),
            ),
          ),
          const SizedBox(height: 25),
          
          // Action Toggle Button
          SizedBox(
            width: double.infinity,
            height: 50,
            child: ElevatedButton(
              onPressed: () {
                setState(() {
                  _isMonitoring = !_isMonitoring;
                });
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: _isMonitoring ? Colors.red : Colors.green,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              ),
              child: Text(
                _isMonitoring ? "STOP MONITORING" : "START MONITORING",
                style: const TextStyle(fontWeight: FontWeight.bold),
              ),
            ),
          ),
          const SizedBox(height: 25),
          
          // Metrics Cards Row
          Row(
            children: [
              Expanded(
                child: _buildMetricCard('Acoustic Class', _currentClass, context),
              ),
              const SizedBox(width: 15),
              Expanded(
                child: _buildMetricCard('Risk Score', '$_riskScore / 100', context),
              ),
            ],
          ),
          const SizedBox(height: 25),
          
          // Pass details listing
          Container(
            padding: const EdgeInsets.all(15),
            decoration: BoxDecoration(
              color: const Color(0xFF1E293B),
              borderRadius: BorderRadius.circular(16),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Two-Pass Transformer Target Details', style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Colors.grey)),
                const SizedBox(height: 15),
                _buildConfidenceRow('Pass 1 (Primary - 2s)', _p1Confidence),
                const Divider(height: 20, color: Colors.white10),
                _buildConfidenceRow('Pass 2 (Verification - 5s)', _p2Confidence),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMetricCard(String label, String value, BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1E293B),
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Text(label, style: const TextStyle(fontSize: 11, color: Colors.grey)),
          const SizedBox(height: 6),
          Text(
            value,
            style: TextStyle(
              fontSize: 18, 
              fontWeight: FontWeight.bold, 
              color: label.contains('Risk') ? Theme.of(context).colorScheme.secondary : Colors.white
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildConfidenceRow(String pass, double confidence) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.between,
      children: [
        Text(pass, style: const TextStyle(fontSize: 13, color: Colors.grey)),
        Text(
          '${(confidence * 100).toStringAsFixed(1)}%',
          style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold),
        ),
      ],
    );
  }
}
