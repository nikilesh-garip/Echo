import 'package:flutter/material.dart';
import 'settings_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _isMonitoring = false;
  bool _mediaPlayback = false;
  bool _suddenMotion = false;

  void _toggleMonitoring() {
    setState(() {
      _isMonitoring = !_isMonitoring;
    });
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(_isMonitoring 
            ? 'Session-based monitoring started' 
            : 'Monitoring session stopped'),
        duration: const Duration(seconds: 2),
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
          Row(
            justifyAxisAlignment: MainAxisAlignment.between,
            children: [
              const Text(
                'ECHO Shield',
                style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold, letterSpacing: 1),
              ),
              IconButton(
                icon: const Icon(Icons.settings),
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => const SettingsScreen()),
                  );
                },
              ),
            ],
          ),
          const SizedBox(height: 10),
          const Text(
            'Keep your environment secure with on-device acoustic hazard intelligence.',
            style: TextStyle(color: Colors.grey, fontSize: 14),
          ),
          const SizedBox(height: 30),
          
          // Monitoring Card
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: const Color(0xFF1E293B),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: Colors.white.withOpacity(0.05)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  justifyAxisAlignment: MainAxisAlignment.between,
                  children: [
                    const Text('Acoustic Sentinel', style: TextStyle(fontSize: 16, color: Colors.grey)),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
                      decoration: BoxDecoration(
                        color: _isMonitoring ? Colors.green.withOpacity(0.15) : Colors.red.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text(
                        _isMonitoring ? 'ACTIVE' : 'INACTIVE',
                        style: TextStyle(
                          color: _isMonitoring ? Colors.green : Colors.red, 
                          fontSize: 11, 
                          fontWeight: FontWeight.bold
                        ),
                      ),
                    )
                  ],
                ),
                const SizedBox(height: 25),
                SizedBox(
                  width: double.infinity,
                  height: 55,
                  child: ElevatedButton(
                    onPressed: _toggleMonitoring,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: _isMonitoring ? Colors.red : const Color(0xFF0D9488),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    ),
                    child: Text(
                      _isMonitoring ? 'STOP SENTINEL' : 'ACTIVATE SENTINEL',
                      style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.white),
                    ),
                  ),
                )
              ],
            ),
          ),
          const SizedBox(height: 30),
          
          // Context Toggles
          const Text('DEVICE CONTEXT SIGNALS', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.grey)),
          const SizedBox(height: 10),
          SwitchListTile(
            title: const Text('Media Audio Active', style: TextStyle(fontSize: 14)),
            subtitle: const Text('Prevents TV/movie sounds from raising risk', style: TextStyle(fontSize: 12, color: Colors.grey)),
            value: _mediaPlayback,
            activeColor: const Color(0xFF0D9488),
            onChanged: (val) {
              setState(() {
                _mediaPlayback = val;
              });
            },
          ),
          SwitchListTile(
            title: const Text('Sudden Motion (Panic)', style: TextStyle(fontSize: 14)),
            subtitle: const Text('Increases risk score if running is detected', style: TextStyle(fontSize: 12, color: Colors.grey)),
            value: _suddenMotion,
            activeColor: const Color(0xFF0D9488),
            onChanged: (val) {
              setState(() {
                _suddenMotion = val;
              });
            },
          ),
        ],
      ),
    );
  }
}
