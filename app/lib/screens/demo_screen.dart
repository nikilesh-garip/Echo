import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import '../services/api_service.dart';
import 'alert_screen.dart';

class DemoScreen extends StatelessWidget {
  DemoScreen({super.key});

  final ApiService _apiService = ApiService(baseUrl: 'http://10.0.2.2:8000');

  final List<String> _demoClasses = const [
    "gunshot", "scream", "glass_breaking", "explosion", "fire_alarm", "siren", "shouting", "normal"
  ];

  Future<void> _triggerSimulatedAlert(BuildContext context, String soundClass) async {
    // Look up guidance rules offline static mapping
    final guidanceMap = {
      "gunshot": {
        "title": "Possible Gunshot Detected",
        "instructions": [
          "Seek substantial cover immediately (concrete walls, thick structures).",
          "Stay low to the ground and avoid exposed windows or doorways.",
          "Silence your phone and call emergency services when absolutely safe to do so."
        ],
        "risk": 85,
        "level": "HIGH_RISK"
      },
      "scream": {
        "title": "Distress Scream Detected",
        "instructions": [
          "Prioritize your personal safety first; do not rush blindly into danger.",
          "Move toward a secure room if you feel threatened.",
          "Call emergency services to report distress coordinate sounds."
        ],
        "risk": 70,
        "level": "POSSIBLE_DANGER"
      }
    };

    final selected = guidanceMap[soundClass] ?? {
      "title": "Possible Threat Detected",
      "instructions": ["Stay alert.", "Monitor local announcements."],
      "risk": 55,
      "level": "SUSPICIOUS"
    };

    double lat = 37.7749; // Default San Francisco coordinates
    double lng = -122.4194;

    try {
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
            print('Geolocator getCurrentPosition failed/timed out in Demo: $timeoutErr');
          }
        }
      }
    } catch (e) {
      print('Error getting GPS location in Demo Screen: $e');
    }

    // Fetch dynamic nearby places from backend OpenStreetMap proxy
    String placeType = (soundClass == "fire_alarm")
        ? "fire"
        : (soundClass == "gunshot" || soundClass == "glass_breaking" || soundClass == "shouting")
            ? "police"
            : "hospital";

    List<Map<String, dynamic>> nearbyFacilities = [];
    try {
      final places = await _apiService.getNearbyPlaces(lat: lat, lng: lng, type: placeType);
      if (places != null) {
        nearbyFacilities = places;
      }
    } catch (e) {
      print('Error fetching nearby places in Demo: $e');
    }

    if (nearbyFacilities.isEmpty) {
      nearbyFacilities = [
        {"name": "Local Emergency Dispatch (GPS Fallback)", "address": "Latitude: ${lat.toStringAsFixed(4)}, Longitude: ${lng.toStringAsFixed(4)}"}
      ];
    }

    // Retrieve contacts to simulate live location message sharing
    List<String> notifiedContacts = [];
    try {
      final contacts = await _apiService.getContacts("echo_mobile_client");
      if (contacts != null && contacts.isNotEmpty) {
        for (var contact in contacts) {
          final contactName = contact['name'] ?? 'Trusted Contact';
          final contactPhone = contact['phone'] ?? '';
          final contactRelation = contact['relation'] ?? 'Friend';
          notifiedContacts.add('$contactName ($contactRelation)');
          print('EMERGENCY SHARING (Demo): Live location link (https://maps.google.com/?q=$lat,$lng) dispatched immediately via SMS to $contactName ($contactPhone)');
        }
      }
    } catch (e) {
      print('Error fetching contacts in Demo: $e');
    }

    if (context.mounted) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => AlertScreen(
            title: selected['title'] as String,
            threatClass: soundClass,
            riskScore: selected['risk'] as int,
            riskLevel: selected['level'] as String,
            p1Conf: 0.95,
            p2Conf: 0.88,
            instructions: List<String>.from(selected['instructions'] as Iterable),
            nearbyFacilities: nearbyFacilities,
            latitude: lat,
            longitude: lng,
            notifiedContacts: notifiedContacts,
          ),
        ),
      );
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
            'Demo & Panel Verification',
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 10),
          const Text(
            'Inject simulated acoustic threat events directly to evaluate guidance rendering, geolocation routing maps, and critical emergency action response interfaces.',
            style: TextStyle(fontSize: 13, color: Colors.grey, height: 1.4),
          ),
          const SizedBox(height: 25),
          
          Expanded(
            child: GridView.builder(
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                crossAxisSpacing: 12,
                mainAxisSpacing: 12,
                childAspectRatio: 1.6,
              ),
              itemCount: _demoClasses.length,
              itemBuilder: (context, index) {
                final cls = _demoClasses[index];
                return InkWell(
                  onTap: () => _triggerSimulatedAlert(context, cls),
                  borderRadius: BorderRadius.circular(12),
                  child: Container(
                    decoration: BoxDecoration(
                      color: const Color(0xFF1E293B),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.white.withOpacity(0.05)),
                    ),
                    child: Center(
                      child: Text(
                        cls.toUpperCase(),
                        style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 13, letterSpacing: 0.5),
                      ),
                    ),
                  ),
                );
              },
            ),
          )
        ],
      ),
    );
  }
}
