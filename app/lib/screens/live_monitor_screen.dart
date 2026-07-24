import 'package:flutter/material.dart';

class LiveMonitorScreen extends StatefulWidget {
  const LiveMonitorScreen({super.key});

  @override
  State<LiveMonitorScreen> createState() => _LiveMonitorScreenState();
}

class _LiveMonitorScreenState extends State<LiveMonitorScreen> {
  String _currentClass = "NORMAL";
  double _p1Confidence = 0.05;
  double _p2Confidence = 0.0;
  int _riskScore = 8;
  String _riskLevel = "NORMAL";

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
          
          // Simulated Waveform Display
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
                'Spectrogram Stream (16 kHz)',
                style: TextStyle(color: Colors.white.withOpacity(0.3), fontSize: 12),
              ),
            ),
          ),
          const SizedBox(height: 25),
          
          // Main metrics row
          Row(
            gap: 15,
            children: [
              Expanded(
                child: _buildMetricCard('Acoustic Class', _currentClass, context),
              ),
              Expanded(
                child: _buildMetricCard('Risk Score', '$_riskScore / 100', context),
              ),
            ],
          ),
          const SizedBox(height: 25),
          
          // Two pass metrics list
          Container(
            padding: const EdgeInsets.all(15),
            decoration: BoxDecoration(
              color: const Color(0xFF1E293B),
              borderRadius: BorderRadius.circular(16),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Two-Pass CRNN Target Details', style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Colors.grey)),
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
