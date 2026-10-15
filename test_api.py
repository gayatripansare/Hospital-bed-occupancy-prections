"""
=============================================================================
CAREPULSE ANALYTICS - RESEARCH ASSISTANT & GRAPHICAL ANALYSIS TEST SUITE
Project 14: Hospital Bed Occupancy Forecasting
=============================================================================
Verifies:
1. Web interface loads CarePulse Research Assistant template (GET /)
2. Question endpoint processes department forecast (POST /api/ask - Cardiology)
3. Bed availability queries return exact numbers (POST /api/ask - Surgery)
4. Peak demand alert detection on surge admissions (POST /api/ask - General Medicine)
5. Contextual follow-up question resolution (POST /api/ask - "What about Neurology?")
6. Benchmark metrics reporting (POST /api/ask - RMSE & R² without model exposure)
7. Latest observed census lookup (POST /api/ask - 2024-12-31 status)
8. Empty question handling and input validation (POST /api/ask - HTTP 400)
9. Graphical analysis metadata catalog endpoint (GET /api/charts-info)
10. Chart A: Historical Bed Occupancy PNG stream (GET /api/chart/historical_occupancy)
11. Chart B: Department-wise Bed Occupancy PNG stream (GET /api/chart/department_occupancy)
12. Chart C: Admissions and Discharges PNG stream (GET /api/chart/admissions_discharges)
13. Chart D: Emergency Cases PNG stream (GET /api/chart/emergency_cases)
14. Chart E: Actual vs Predicted Occupancy PNG stream (GET /api/chart/actual_vs_predicted)
=============================================================================
"""

import sys
import time
import json
from app import app

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_separator():
    print("-" * 75)

def run_tests():
    print(f"\n{BOLD}{CYAN}==========================================================================={RESET}")
    print(f"{BOLD}{CYAN}      CAREPULSE ANALYTICS - RESEARCH ASSISTANT & CHARTS TEST SUITE           {RESET}")
    print(f"{BOLD}{CYAN}==========================================================================={RESET}\n")

    client = app.test_client()
    passed_count = 0

    tests = [
        {
            "name": "Test 1: Web Interface with Left Sidebar & Dual Screens (GET /)",
            "method": "GET",
            "url": "/",
            "payload": None,
            "validate": lambda res: (
                res.status_code == 200 and
                b"CarePulse Research Assistant" in res.data and
                b"Graphical Analysis" in res.data and
                b"sidebar" in res.data
            ),
            "description": "Verifies that the dual-screen sidebar interface loads correctly."
        },
        {
            "name": "Test 2: Department Occupancy Forecast (POST /api/ask - Cardiology)",
            "method": "POST",
            "url": "/api/ask",
            "payload": {"question": "Forecast bed occupancy for Cardiology tomorrow"},
            "validate": lambda res: (
                res.status_code == 200 and
                res.get_json().get("success") is True and
                res.get_json().get("metrics") is not None and
                res.get_json()["metrics"].get("department") == "Cardiology" and
                isinstance(res.get_json()["metrics"].get("predicted_occupancy_rate"), (int, float))
            ),
            "description": "Validates ML model inference for continuous bed occupancy."
        },
        {
            "name": "Test 3: Bed Availability Query (POST /api/ask - Surgery)",
            "method": "POST",
            "url": "/api/ask",
            "payload": {"question": "How many beds are available in Surgery?"},
            "validate": lambda res: (
                res.status_code == 200 and
                res.get_json().get("success") is True and
                res.get_json()["metrics"].get("department") == "Surgery" and
                "estimated_available_beds" in res.get_json()["metrics"]
            ),
            "description": "Checks that remaining bed availability is calculated correctly."
        },
        {
            "name": "Test 4: Peak Demand Stress Detection (POST /api/ask - 20 Admissions)",
            "method": "POST",
            "url": "/api/ask",
            "payload": {"question": "Will General Medicine face peak demand with 20 admissions and 8 emergencies?"},
            "validate": lambda res: (
                res.status_code == 200 and
                res.get_json().get("success") is True and
                res.get_json()["metrics"].get("predicted_occupancy_rate") >= 85.0 and
                res.get_json()["metrics"].get("is_peak_demand") is True
            ),
            "description": "Verifies that high patient admissions trigger the >= 85% peak demand alert."
        },
        {
            "name": "Test 5: Conversational Follow-Up Resolution (POST /api/ask - Context Memory)",
            "method": "POST",
            "url": "/api/ask",
            "payload": {
                "question": "What about Neurology?",
                "history": [
                    {"role": "user", "content": "Forecast bed occupancy for Cardiology with 12 admissions"}
                ]
            },
            "validate": lambda res: (
                res.status_code == 200 and
                res.get_json().get("success") is True and
                res.get_json()["metrics"].get("department") == "Neurology"
            ),
            "description": "Ensures follow-up questions resolve correctly using conversational context."
        },
        {
            "name": "Test 6: Model Evaluation Benchmark (POST /api/ask - Anonymized Metrics)",
            "method": "POST",
            "url": "/api/ask",
            "payload": {"question": "What is the model evaluation RMSE and R2 score?"},
            "validate": lambda res: (
                res.status_code == 200 and
                res.get_json().get("success") is True and
                "2.231" in res.get_json().get("answer", "") and
                "0.976" in res.get_json().get("answer", "") and
                "Linear Regression" not in res.get_json().get("answer", "")
            ),
            "description": "Verifies accuracy metrics are reported without exposing model names."
        },
        {
            "name": "Test 7: Latest Observed Census Inquiry (POST /api/ask - Hospital Overview)",
            "method": "POST",
            "url": "/api/ask",
            "payload": {"question": "What was the latest observed hospital census on December 31?"},
            "validate": lambda res: (
                res.status_code == 200 and
                res.get_json().get("success") is True and
                "536 beds" in res.get_json().get("answer", "")
            ),
            "description": "Validates actual historical observation retrieval from hospital_occupancy.csv."
        },
        {
            "name": "Test 8: Empty Question Validation (POST /api/ask - Bad Request)",
            "method": "POST",
            "url": "/api/ask",
            "payload": {"question": ""},
            "validate": lambda res: (
                res.status_code == 400 and
                res.get_json().get("success") is False
            ),
            "description": "Ensures empty inputs return HTTP 400 with a clear error message."
        },
        {
            "name": "Test 9: Graphical Analysis Catalog (GET /api/charts-info)",
            "method": "GET",
            "url": "/api/charts-info",
            "payload": None,
            "validate": lambda res: (
                res.status_code == 200 and
                res.get_json().get("success") is True and
                len(res.get_json().get("charts", [])) == 5
            ),
            "description": "Verifies that chart metadata for all 5 Matplotlib graphs is delivered."
        },
        {
            "name": "Test 10: Matplotlib Graph A - Historical Bed Occupancy (GET /api/chart/historical_occupancy)",
            "method": "GET",
            "url": "/api/chart/historical_occupancy",
            "payload": None,
            "validate": lambda res: (
                res.status_code == 200 and
                res.mimetype == "image/png" and
                len(res.data) > 5000
            ),
            "description": "Ensures Graph A generates a valid high-resolution PNG image."
        },
        {
            "name": "Test 11: Matplotlib Graph B - Department-wise Bed Occupancy (GET /api/chart/department_occupancy)",
            "method": "GET",
            "url": "/api/chart/department_occupancy",
            "payload": None,
            "validate": lambda res: (
                res.status_code == 200 and
                res.mimetype == "image/png" and
                len(res.data) > 5000
            ),
            "description": "Ensures Graph B generates a valid high-resolution PNG image."
        },
        {
            "name": "Test 12: Matplotlib Graph C - Admissions & Discharges (GET /api/chart/admissions_discharges)",
            "method": "GET",
            "url": "/api/chart/admissions_discharges",
            "payload": None,
            "validate": lambda res: (
                res.status_code == 200 and
                res.mimetype == "image/png" and
                len(res.data) > 5000
            ),
            "description": "Ensures Graph C generates a valid high-resolution PNG image."
        },
        {
            "name": "Test 13: Matplotlib Graph D - Emergency Cases Trend (GET /api/chart/emergency_cases)",
            "method": "GET",
            "url": "/api/chart/emergency_cases",
            "payload": None,
            "validate": lambda res: (
                res.status_code == 200 and
                res.mimetype == "image/png" and
                len(res.data) > 5000
            ),
            "description": "Ensures Graph D generates a valid high-resolution PNG image."
        },
        {
            "name": "Test 14: Matplotlib Graph E - Actual vs Predicted Validation (GET /api/chart/actual_vs_predicted)",
            "method": "GET",
            "url": "/api/chart/actual_vs_predicted",
            "payload": None,
            "validate": lambda res: (
                res.status_code == 200 and
                res.mimetype == "image/png" and
                len(res.data) > 5000
            ),
            "description": "Ensures Graph E generates a valid high-resolution PNG image from test data."
        }
    ]

    total_tests = len(tests)

    for idx, test in enumerate(tests, 1):
        print_separator()
        print(f"{BOLD}[{idx}/{total_tests}] Running: {test['name']}{RESET}")
        print(f"  Description : {test['description']}")
        print(f"  Method/Route: {BOLD}{test['method']} {test['url']}{RESET}")

        start_time = time.time()
        try:
            if test['method'] == 'GET':
                response = client.get(test['url'])
            else:
                response = client.post(test['url'], json=test['payload'])

            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            is_valid = test['validate'](response)

            if is_valid:
                passed_count += 1
                status_str = f"{GREEN}{BOLD}PASS{RESET}"
            else:
                status_str = f"{RED}{BOLD}FAIL{RESET}"

            print(f"  Status Code : {response.status_code}")
            print(f"  Latency     : {elapsed_ms} ms")
            print(f"  Result      : {status_str}")

            if response.is_json:
                data = response.get_json()
                if "metrics" in data and data["metrics"]:
                    m = data["metrics"]
                    print(f"  Output      : Dept = {BOLD}{m.get('department')}{RESET} | Rate = {BOLD}{m.get('predicted_occupancy_rate')}%{RESET} | Available = {BOLD}{m.get('estimated_available_beds')} beds{RESET} | Peak = {BOLD}{m.get('is_peak_demand')}{RESET}")
                elif "charts" in data:
                    print(f"  Charts Count: {len(data['charts'])} visualizations loaded.")
                elif "answer" in data:
                    print(f"  Answer Snippet: {data['answer'][:80].replace(chr(10), ' ')}...")
            elif response.mimetype == "image/png":
                print(f"  Image Data  : {len(response.data):,} bytes (image/png)")

        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            print(f"  Latency     : {elapsed_ms} ms")
            print(f"  Result      : {RED}{BOLD}ERROR ({str(e)}){RESET}")

    print_separator()
    print(f"\n{BOLD}{CYAN}==========================================================================={RESET}")
    print(f"{BOLD} TEST SUMMARY: {GREEN if passed_count == total_tests else RED}{passed_count} / {total_tests} Tests Passed ({(passed_count/total_tests)*100:.1f}% Success Rate){RESET}")
    print(f"{BOLD}{CYAN}==========================================================================={RESET}\n")

    return passed_count == total_tests

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
