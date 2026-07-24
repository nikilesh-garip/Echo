import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  final String baseUrl;
  
  ApiService({required this.baseUrl});
  
  /// Sends a 2-second or 5-second audio chunk to the backend for threat detection.
  Future<Map<String, dynamic>?> detectAudio({
    required File audioFile,
    required double duration,
    bool mediaPlayback = false,
    bool suddenMotion = false,
  }) async {
    try {
      final uri = Uri.parse('$baseUrl/detect');
      final request = http.MultipartRequest('POST', uri)
        ..fields['duration'] = duration.toString()
        ..fields['media_playback'] = mediaPlayback.toString()
        ..fields['sudden_motion'] = suddenMotion.toString()
        ..files.add(await http.MultipartFile.fromPath(
          'file',
          audioFile.path,
        ));
        
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);
      
      if (response.statusCode == 200) {
        return json.decode(response.body) as Map<String, dynamic>;
      } else {
        print('Detect request failed: ${response.statusCode}');
        return null;
      }
    } catch (e) {
      print('Error calling detect endpoint: $e');
      return null;
    }
  }

  /// Logs a verified emergency event to the backend.
  Future<bool> logEvent({
    required String userId,
    required String className,
    required double primaryConf,
    required double verificationConf,
    required int riskScore,
    required String riskLevel,
  }) async {
    try {
      final uri = Uri.parse('$baseUrl/events');
      final response = await http.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'user_id': userId,
          'class_name': className,
          'primary_conf': primaryConf,
          'verification_conf': verificationConf,
          'risk_score': riskScore,
          'risk_level': riskLevel,
        }),
      );
      return response.statusCode == 200;
    } catch (e) {
      print('Error logging event: $e');
      return false;
    }
  }

  /// Fetches contact list from backend database.
  Future<List<Map<String, dynamic>>?> getContacts(String userId) async {
    try {
      final uri = Uri.parse('$baseUrl/contacts/$userId');
      final response = await http.get(uri);
      if (response.statusCode == 200) {
        return List<Map<String, dynamic>>.from(json.decode(response.body));
      }
      return null;
    } catch (e) {
      print('Error fetching contacts: $e');
      return null;
    }
  }

  /// Saves a new emergency contact to the database.
  Future<Map<String, dynamic>?> addContact({
    required String userId,
    required String name,
    required String phone,
    required String relation,
  }) async {
    try {
      final uri = Uri.parse('$baseUrl/contacts');
      final response = await http.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'user_id': userId,
          'name': name,
          'phone': phone,
          'relation': relation,
        }),
      );
      if (response.statusCode == 200) {
        return json.decode(response.body) as Map<String, dynamic>;
      }
      return null;
    } catch (e) {
      print('Error adding contact: $e');
      return null;
    }
  }

  /// Deletes a contact from the backend.
  Future<bool> deleteContact(int contactId) async {
    try {
      final uri = Uri.parse('$baseUrl/contacts/$contactId');
      final response = await http.delete(uri);
      return response.statusCode == 200;
    } catch (e) {
      print('Error deleting contact: $e');
      return false;
    }
  }

  /// Fetches nearby places (hospitals, police stations, fire departments) using OSM Proxy.
  Future<List<Map<String, dynamic>>?> getNearbyPlaces({
    required double lat,
    required double lng,
    required String type,
  }) async {
    try {
      final uri = Uri.parse('$baseUrl/nearby?lat=$lat&lng=$lng&type=$type');
      final response = await http.get(uri);
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body) as Map<String, dynamic>;
        if (data['status'] == 'success' || data['status'] == 'fallback') {
          return List<Map<String, dynamic>>.from(data['results']);
        }
      }
      return null;
    } catch (e) {
      print('Error fetching nearby places: $e');
      return null;
    }
  }
}
