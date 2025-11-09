import requests
import sys
import json
import base64
from datetime import datetime
import time

class CAMATSAPITester:
    def __init__(self, base_url="https://cam-ats.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.token = None
        self.user_id = None
        self.session_id = None
        self.tests_run = 0
        self.tests_passed = 0
        self.test_user_email = f"test_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}@test.com"
        self.test_user_password = "TestPass123!"
        self.test_user_name = "Test User"

    def log_test(self, name, success, details=""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED {details}")
        else:
            print(f"❌ {name} - FAILED {details}")
        return success

    def make_request(self, method, endpoint, data=None, expected_status=200):
        """Make HTTP request with proper headers"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'

        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=30)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, timeout=30)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=30)

            success = response.status_code == expected_status
            return success, response.status_code, response.json() if response.content else {}

        except requests.exceptions.RequestException as e:
            return False, 0, {"error": str(e)}
        except json.JSONDecodeError:
            return False, response.status_code, {"error": "Invalid JSON response"}

    def test_health_check(self):
        """Test API health endpoint"""
        success, status, response = self.make_request('GET', 'health', expected_status=200)
        return self.log_test("Health Check", success, f"Status: {status}")

    def test_root_endpoint(self):
        """Test root API endpoint"""
        success, status, response = self.make_request('GET', '', expected_status=200)
        return self.log_test("Root Endpoint", success, f"Status: {status}")

    def test_user_registration(self):
        """Test user registration"""
        user_data = {
            "email": self.test_user_email,
            "name": self.test_user_name,
            "password": self.test_user_password
        }
        
        success, status, response = self.make_request('POST', 'auth/register', user_data, expected_status=200)
        
        if success and 'access_token' in response:
            self.token = response['access_token']
            self.user_id = response['user']['id']
            return self.log_test("User Registration", True, f"User ID: {self.user_id}")
        else:
            return self.log_test("User Registration", False, f"Status: {status}, Response: {response}")

    def test_user_login(self):
        """Test user login"""
        login_data = {
            "email": self.test_user_email,
            "password": self.test_user_password
        }
        
        success, status, response = self.make_request('POST', 'auth/login', login_data, expected_status=200)
        
        if success and 'access_token' in response:
            self.token = response['access_token']
            return self.log_test("User Login", True, f"Token received")
        else:
            return self.log_test("User Login", False, f"Status: {status}, Response: {response}")

    def test_dashboard_data(self):
        """Test dashboard data retrieval"""
        success, status, response = self.make_request('GET', 'dashboard', expected_status=200)
        
        if success and 'user' in response and 'statistics' in response:
            return self.log_test("Dashboard Data", True, f"User: {response['user']['name']}")
        else:
            return self.log_test("Dashboard Data", False, f"Status: {status}, Response: {response}")

    def test_create_session(self):
        """Test session creation"""
        session_data = {
            "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "title": "Test Learning Session"
        }
        
        success, status, response = self.make_request('POST', 'sessions', session_data, expected_status=200)
        
        if success and 'id' in response:
            self.session_id = response['id']
            return self.log_test("Create Session", True, f"Session ID: {self.session_id}")
        else:
            return self.log_test("Create Session", False, f"Status: {status}, Response: {response}")

    def test_get_sessions(self):
        """Test getting user sessions"""
        success, status, response = self.make_request('GET', 'sessions', expected_status=200)
        
        if success and isinstance(response, list):
            return self.log_test("Get Sessions", True, f"Found {len(response)} sessions")
        else:
            return self.log_test("Get Sessions", False, f"Status: {status}, Response: {response}")

    def test_get_specific_session(self):
        """Test getting a specific session"""
        if not self.session_id:
            return self.log_test("Get Specific Session", False, "No session ID available")
        
        success, status, response = self.make_request('GET', f'sessions/{self.session_id}', expected_status=200)
        
        if success and response.get('id') == self.session_id:
            return self.log_test("Get Specific Session", True, f"Session: {response.get('title')}")
        else:
            return self.log_test("Get Specific Session", False, f"Status: {status}, Response: {response}")

    def test_image_analysis(self):
        """Test image analysis endpoint with a simple test image"""
        if not self.session_id:
            return self.log_test("Image Analysis", False, "No session ID available")
        
        # Create a simple test image (1x1 pixel base64 encoded)
        test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        analysis_data = {
            "session_id": self.session_id,
            "image_data": f"data:image/png;base64,{test_image_b64}"
        }
        
        success, status, response = self.make_request('POST', 'analyze/image', analysis_data, expected_status=200)
        
        if success and 'analysis' in response:
            return self.log_test("Image Analysis", True, f"Analysis completed")
        else:
            return self.log_test("Image Analysis", False, f"Status: {status}, Response: {response}")

    def test_tab_switch_reporting(self):
        """Test tab switch reporting"""
        if not self.session_id:
            return self.log_test("Tab Switch Reporting", False, "No session ID available")
        
        success, status, response = self.make_request('POST', f'sessions/{self.session_id}/tab-switch', expected_status=200)
        
        if success and 'alert_id' in response:
            return self.log_test("Tab Switch Reporting", True, f"Alert ID: {response['alert_id']}")
        else:
            return self.log_test("Tab Switch Reporting", False, f"Status: {status}, Response: {response}")

    def test_get_session_alerts(self):
        """Test getting session alerts"""
        if not self.session_id:
            return self.log_test("Get Session Alerts", False, "No session ID available")
        
        success, status, response = self.make_request('GET', f'sessions/{self.session_id}/alerts', expected_status=200)
        
        if success and isinstance(response, list):
            return self.log_test("Get Session Alerts", True, f"Found {len(response)} alerts")
        else:
            return self.log_test("Get Session Alerts", False, f"Status: {status}, Response: {response}")

    def test_get_session_analytics(self):
        """Test getting session analytics"""
        if not self.session_id:
            return self.log_test("Get Session Analytics", False, "No session ID available")
        
        success, status, response = self.make_request('GET', f'sessions/{self.session_id}/analytics', expected_status=200)
        
        if success and 'session' in response and 'alert_statistics' in response:
            return self.log_test("Get Session Analytics", True, f"Analytics retrieved")
        else:
            return self.log_test("Get Session Analytics", False, f"Status: {status}, Response: {response}")

    def test_end_session(self):
        """Test ending a session"""
        if not self.session_id:
            return self.log_test("End Session", False, "No session ID available")
        
        success, status, response = self.make_request('POST', f'sessions/{self.session_id}/end', expected_status=200)
        
        if success and 'final_score' in response:
            return self.log_test("End Session", True, f"Final score: {response['final_score']}")
        else:
            return self.log_test("End Session", False, f"Status: {status}, Response: {response}")

    def run_all_tests(self):
        """Run all API tests in sequence"""
        print("🚀 Starting CAM ATS API Testing...")
        print(f"📍 Testing against: {self.base_url}")
        print("=" * 60)
        
        # Basic connectivity tests
        self.test_health_check()
        self.test_root_endpoint()
        
        # Authentication tests
        self.test_user_registration()
        self.test_user_login()
        
        # Dashboard test
        self.test_dashboard_data()
        
        # Session management tests
        self.test_create_session()
        self.test_get_sessions()
        self.test_get_specific_session()
        
        # Analysis and monitoring tests
        self.test_image_analysis()
        self.test_tab_switch_reporting()
        self.test_get_session_alerts()
        self.test_get_session_analytics()
        
        # Session completion test
        self.test_end_session()
        
        # Print final results
        print("=" * 60)
        print(f"📊 Test Results: {self.tests_passed}/{self.tests_run} tests passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed! Backend API is working correctly.")
            return True
        else:
            print(f"⚠️  {self.tests_run - self.tests_passed} tests failed. Backend needs attention.")
            return False

def main():
    """Main test execution"""
    tester = CAMATSAPITester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())