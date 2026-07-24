import 'package:flutter/material.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0F172A),
      appBar: AppBar(
        title: const Text('Settings'),
        backgroundColor: const Color(0xFF1E293B),
      ),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          // Sensitivity Settings
          _buildSectionHeader('Detection Settings'),
          ListTile(
            title: const Text('Acoustic Sensitivity'),
            subtitle: const Text('Threshold configuration: Medium (0.50)'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () {},
          ),
          const Divider(height: 1, color: Colors.white10),
          
          // Privacy
          const SizedBox(height: 20),
          _buildSectionHeader('Privacy & Permissions'),
          const ListTile(
            title: Text('Microphone Permission'),
            subtitle: Text('Authorized for foreground session buffers'),
            trailing: Icon(Icons.check_circle, color: Colors.green),
          ),
          const Divider(height: 1, color: Colors.white10),
          const ListTile(
            title: Text('Location Permission'),
            subtitle: Text('Authorized for emergency dispatch routing'),
            trailing: Icon(Icons.check_circle, color: Colors.green),
          ),
          const Divider(height: 1, color: Colors.white10),
          
          // System details
          const SizedBox(height: 20),
          _buildSectionHeader('About System'),
          const ListTile(
            title: Text('Model Engine'),
            subtitle: Text('ONNX Runtime / TFLite Quantized'),
          ),
          const Divider(height: 1, color: Colors.white10),
          const ListTile(
            title: Text('Model Checkpoint Version'),
            subtitle: Text('Echo CRNN v1.0.0 (Fixed Shape)'),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 16.0),
      child: Text(
        title.toUpperCase(),
        style: const TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.bold,
          color: Colors.grey,
          letterSpacing: 0.5,
        ),
      ),
    );
  }
}
