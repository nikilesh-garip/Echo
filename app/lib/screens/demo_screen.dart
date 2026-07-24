import 'package:flutter/material.dart';
import 'alert_screen.dart';

class DemoScreen extends StatelessWidget {
  const DemoScreen({super.key});

  final List<String> _demoClasses = const [
    "gunshot", "scream", "glass_breaking", "explosion", "fire_alarm", "siren", "shouting", "normal"
  ];

  void _triggerSimulatedAlert(BuildContext context, String soundClass) {
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
          nearbyFacilities: const [
            {"name": "General District Police Station", "address": "456 Safety Blvd"},
            {"name": "City Medical Center Emergency", "address": "123 Civic Center Way"}
          ],
        ),
      ),
    );
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
