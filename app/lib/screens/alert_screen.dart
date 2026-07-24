import 'package:flutter/material.dart';

class AlertScreen extends StatelessWidget {
  final String title;
  final String threatClass;
  final int riskScore;
  final String riskLevel;
  final double p1Conf;
  final double p2Conf;
  final List<String> instructions;
  final List<Map<String, dynamic>> nearbyFacilities;

  const AlertScreen({
    super.key,
    required this.title,
    required this.threatClass,
    required this.riskScore,
    required this.riskLevel,
    required this.p1Conf,
    required this.p2Conf,
    required this.instructions,
    required this.nearbyFacilities,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0F172A),
      body: SafeArea(
        child: Column(
          children: [
            // Alert Header
            Container(
              padding: const EdgeInsets.symmetric(vertical: 30, horizontal: 20),
              decoration: const BoxDecoration(
                color: Color(0xFF7F1D1D), // Dark Red
                borderRadius: BorderRadius.only(
                  bottomLeft: Radius.circular(30),
                  bottomRight: Radius.circular(30),
                ),
              ),
              child: Column(
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: const Text(
                      'CRITICAL ALERT',
                      style: TextStyle(color: Color(0xFFEF4444), fontSize: 11, fontWeight: FontWeight.bold),
                    ),
                  ),
                  const SizedBox(height: 15),
                  Text(
                    title,
                    style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.white),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 10),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Text('Risk Score: ', style: TextStyle(color: Colors.white70)),
                      Text(
                        '$riskScore ($riskLevel)',
                        style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.white),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            
            // Body Content
            Expanded(
              child: ListView(
                padding: const EdgeInsets.all(20),
                children: [
                  // Inference metrics
                  _buildSectionTitle('Inference Details'),
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1E293B),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text('Pass 1 (Primary): ${(p1Conf * 100).toStringAsFixed(0)}%', style: const TextStyle(fontSize: 12)),
                        Text('Pass 2 (Verify): ${(p2Conf * 100).toStringAsFixed(0)}%', style: const TextStyle(fontSize: 12)),
                      ],
                    ),
                  ),
                  const SizedBox(height: 25),
                  
                  // Emergency Guidance
                  _buildSectionTitle('Emergency Action Plan'),
                  ...instructions.map((step) => Padding(
                    padding: const EdgeInsets.symmetric(vertical: 6.0),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Icon(Icons.warning_amber, color: Color(0xFFF97316), size: 18),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Text(
                            step,
                            style: const TextStyle(fontSize: 13, color: Colors.white70, height: 1.4),
                          ),
                        ),
                      ],
                    ),
                  )),
                  const SizedBox(height: 25),
                  
                  // Facilities Map lookup
                  _buildSectionTitle('Nearby Public Services'),
                  ...nearbyFacilities.map((facility) => Card(
                    color: const Color(0xFF1E293B),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    child: ListTile(
                      title: Text(facility['name'], style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold)),
                      subtitle: Text(facility['address'], style: const TextStyle(fontSize: 11, color: Colors.grey)),
                      trailing: const Icon(Icons.directions, color: Color(0xFF0D9488)),
                      onTap: () {},
                    ),
                  )),
                ],
              ),
            ),
            
            // Bottom Action buttons
            Padding(
              padding: const EdgeInsets.all(20.0),
              child: Row(
                children: [
                  Expanded(
                    child: SizedBox(
                      height: 50,
                      child: ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFFEF4444),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                        ),
                        onPressed: () {},
                        child: const Text('Call 911 / Services', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                      ),
                    ),
                  ),
                  const SizedBox(width: 15),
                  Expanded(
                    child: SizedBox(
                      height: 50,
                      child: OutlinedButton(
                        style: OutlinedButton.styleFrom(
                          side: const BorderSide(color: Colors.grey),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                        ),
                        onPressed: () => Navigator.pop(context),
                        child: const Text('I\'m Safe', style: TextStyle(color: Colors.white)),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10.0),
      style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.grey, letterSpacing: 0.5),
      child: Text(title.toUpperCase()),
    );
  }
}
