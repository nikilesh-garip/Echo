import 'package:flutter/material.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  final List<Map<String, dynamic>> _events = [
    {
      "class_name": "gunshot",
      "timestamp": "23-07-2026 19:15",
      "risk_score": 85,
      "risk_level": "HIGH_RISK"
    },
    {
      "class_name": "scream",
      "timestamp": "23-07-2026 19:12",
      "risk_score": 70,
      "risk_level": "POSSIBLE_DANGER"
    }
  ];

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(20.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Event Log History',
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 5),
          const Text(
            'Echo maintains metadata-only history. Raw continuous audio is never stored.',
            style: TextStyle(fontSize: 12, color: Colors.grey),
          ),
          const SizedBox(height: 20),
          
          Expanded(
            child: _events.isEmpty
                ? const Center(child: Text('No events logged.'))
                : ListView.builder(
                    itemCount: _events.length,
                    itemBuilder: (context, index) {
                      final item = _events[index];
                      final isHigh = item['risk_level'] == 'HIGH_RISK';
                      
                      return Card(
                        color: const Color(0xFF1E293B),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                          side: BorderSide(
                            color: isHigh ? Colors.red.withOpacity(0.3) : Colors.orange.withOpacity(0.3),
                            width: 1
                          )
                        ),
                        margin: const EdgeInsets.only(bottom: 12),
                        child: ListTile(
                          title: Text(
                            item['class_name'].toString().toUpperCase(),
                            style: const TextStyle(fontWeight: FontWeight.bold),
                          ),
                          subtitle: Text(
                            item['timestamp'],
                            style: const TextStyle(fontSize: 11, color: Colors.grey),
                          ),
                          trailing: Text(
                            'Risk: ${item['risk_score']}',
                            style: TextStyle(
                              color: isHigh ? Colors.red : Colors.orange, 
                              fontWeight: FontWeight.bold
                            ),
                          ),
                        ),
                      );
                    },
                  ),
          ),
          const SizedBox(height: 10),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton(
              onPressed: () {
                setState(() {
                  _events.clear();
                });
              },
              child: const Text('Clear All Metadata', style: TextStyle(color: Colors.white)),
            ),
          )
        ],
      ),
    );
  }
}
