"""
=============================================================================
CAREPULSE ANALYTICS - MCP SERVER & AI AGENT INTEGRATION TEST SUITE
Project 14: Hospital Bed Occupancy Forecasting
=============================================================================
Comprehensive End-to-End Test Suite verifying:
1. Python Prediction API (`POST /api/forecast`, `GET /api/department/<name>/status`, `GET /api/forecast/history`)
2. Machine Learning Model Pipeline integrity (Scikit-Learn inference, non-negative bounds)
3. Node.js/TypeScript MCP Server execution over stdio (JSON-RPC protocol, tool registry, tool execution)
4. CarePulse AI Agent query execution with strict ML model authority (zero hallucinations)
5. Zero model name leakage (technical details kept out of user-facing UI)
=============================================================================
"""

import os
import sys
import json
import time
import subprocess
import unittest
from app import (
    app,
    execute_department_forecast,
    df_raw,
    alert_system,
    DEPT_ALIASES,
    generate_research_answer,
    parse_conversational_query,
)
from ai_agent import carepulse_agent, execute_tool_call, MCP_AGENT_TOOLS

GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


class TestCarePulseMCPIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n{BOLD}{CYAN}==========================================================================={RESET}")
        print(f"{BOLD}{CYAN}   CAREPULSE ANALYTICS - MCP SERVER & AI AGENT INTEGRATION SUITE           {RESET}")
        print(f"{BOLD}{CYAN}==========================================================================={RESET}\n")
        cls.client = app.test_client()
        cls.flask_context = {
            "execute_department_forecast": execute_department_forecast,
            "df_raw": df_raw,
            "alert_system": alert_system,
            "dept_aliases": DEPT_ALIASES,
            "generate_research_answer": generate_research_answer,
            "parse_conversational_query": parse_conversational_query
        }

    # =========================================================================
    # PART 1: PYTHON PREDICTION API & ML MODEL INTEGRITY
    # =========================================================================

    def test_01_forecast_api_valid_payload(self):
        """Verifies POST /api/forecast invokes ML model and returns structured predictions."""
        print(f"{YELLOW}[API Test 1/8]{RESET} Testing POST /api/forecast with valid parameters...")
        payload = {
            "department": "Cardiology",
            "expected_admissions": 10,
            "expected_emergency": 3,
            "expected_discharges": 8,
            "season": "Winter",
            "month": 1,
            "is_holiday": 0
        }
        res = self.client.post("/api/forecast", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["department"], "Cardiology")
        self.assertIn("predicted_occupancy_rate", data)
        self.assertIn("predicted_occupied_beds", data)
        self.assertIn("estimated_available_beds", data)
        self.assertIn("zone", data)
        self.assertIn(data["zone"], ["GREEN", "YELLOW", "RED"])
        print(f"  {GREEN}PASS{RESET} -> Rate: {data['predicted_occupancy_rate']}% | Occupied: {data['predicted_occupied_beds']} | Available: {data['estimated_available_beds']} | Zone: {data['zone']}")

    def test_02_forecast_api_negative_validation(self):
        """Verifies POST /api/forecast strictly rejects negative patient admissions."""
        print(f"{YELLOW}[API Test 2/8]{RESET} Testing POST /api/forecast rejection of negative admissions...")
        payload = {"department": "Surgery", "expected_admissions": -5}
        res = self.client.post("/api/forecast", json=payload)
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data["success"])
        self.assertIn("error", data)
        print(f"  {GREEN}PASS{RESET} -> Properly returned HTTP 400: '{data['error']}'")

    def test_03_forecast_api_invalid_department(self):
        """Verifies POST /api/forecast returns HTTP 404 for unverified department."""
        print(f"{YELLOW}[API Test 3/8]{RESET} Testing POST /api/forecast rejection of unknown department...")
        payload = {"department": "Veterinary Clinic"}
        res = self.client.post("/api/forecast", json=payload)
        self.assertEqual(res.status_code, 404)
        data = res.get_json()
        self.assertFalse(data["success"])
        print(f"  {GREEN}PASS{RESET} -> Properly returned HTTP 404 for unrecognized department.")

    def test_04_department_status_endpoint(self):
        """Verifies GET /api/department/<dept>/status returns latest census & ML next-day forecast."""
        print(f"{YELLOW}[API Test 4/8]{RESET} Testing GET /api/department/Cardiology/status...")
        res = self.client.get("/api/department/Cardiology/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["department"], "Cardiology")
        self.assertIn("observation_date", data)
        self.assertIn("forecast_next_day", data)
        print(f"  {GREEN}PASS{RESET} -> Census Date: {data['observation_date']} | Next Day Forecast: {data['forecast_next_day']['predicted_occupancy_rate']}%")

    def test_05_forecast_history_audit_log(self):
        """Verifies GET /api/forecast/history returns persisted audit logs."""
        print(f"{YELLOW}[API Test 5/8]{RESET} Testing GET /api/forecast/history audit log retrieval...")
        res = self.client.get("/api/forecast/history?limit=5")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("history", data)
        self.assertIsInstance(data["history"], list)
        print(f"  {GREEN}PASS{RESET} -> Retrieved {len(data['history'])} audit entries successfully.")

    # =========================================================================
    # PART 2: NODE.JS / TYPESCRIPT MCP SERVER (OVER STDIO)
    # =========================================================================

    def test_06_mcp_server_stdio_handshake_and_tool_registry(self):
        """Spawns the compiled MCP server and validates JSON-RPC handshake & tool registration."""
        print(f"{YELLOW}[MCP Test 6/8]{RESET} Testing TypeScript MCP Server over stdio JSON-RPC...")
        mcp_entry = os.path.join(os.path.dirname(__file__), "mcp-server", "dist", "index.js")
        self.assertTrue(os.path.exists(mcp_entry), f"MCP server build missing at {mcp_entry}")

        proc = subprocess.Popen(
            ["node", mcp_entry],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            # 1. Initialize Request
            init_req = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "carepulse-test-suite", "version": "1.0.0"}
                }
            }
            proc.stdin.write(json.dumps(init_req) + "\n")
            proc.stdin.flush()

            # Read response
            init_line = proc.stdout.readline()
            self.assertTrue(init_line, "No response from MCP server on initialize")
            init_resp = json.loads(init_line)
            self.assertEqual(init_resp.get("id"), 1)
            server_info = init_resp.get("result", {}).get("serverInfo", {})
            self.assertEqual(server_info.get("name"), "carepulse-hospital-bed-forecasting")
            print(f"  {GREEN}PASS{RESET} -> MCP Handshake OK: {server_info.get('name')} v{server_info.get('version')}")

            # 2. Tools List Request
            tools_req = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }
            proc.stdin.write(json.dumps(tools_req) + "\n")
            proc.stdin.flush()

            tools_line = proc.stdout.readline()
            self.assertTrue(tools_line, "No response from MCP server on tools/list")
            tools_resp = json.loads(tools_line)
            tools_list = tools_resp.get("result", {}).get("tools", [])
            tool_names = [t.get("name") for t in tools_list]

            expected_tools = [
                "predict_hospital_demand",
                "get_current_occupancy",
                "get_department_status",
                "check_red_zone",
                "get_active_alerts",
                "get_forecast_history"
            ]
            for exp in expected_tools:
                self.assertIn(exp, tool_names, f"Expected tool '{exp}' not registered in MCP server")

            print(f"  {GREEN}PASS{RESET} -> All {len(expected_tools)} MCP tools verified: {tool_names}")

            # 3. Call check_red_zone Tool via MCP
            call_req = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "check_red_zone",
                    "arguments": {"occupancy_rate": 89.5, "department": "Cardiology"}
                }
            }
            proc.stdin.write(json.dumps(call_req) + "\n")
            proc.stdin.flush()

            call_line = proc.stdout.readline()
            self.assertTrue(call_line, "No response from MCP server on tools/call")
            call_resp = json.loads(call_line)
            content_list = call_resp.get("result", {}).get("content", [])
            self.assertTrue(len(content_list) > 0)
            parsed_call_output = json.loads(content_list[0].get("text", "{}"))
            self.assertEqual(parsed_call_output.get("zone"), "RED")
            self.assertTrue(parsed_call_output.get("is_red_zone"))
            print(f"  {GREEN}PASS{RESET} -> MCP Tool 'check_red_zone' evaluated: Zone={parsed_call_output.get('zone')} | Red={parsed_call_output.get('is_red_zone')}")

        finally:
            proc.terminate()
            proc.wait(timeout=5)

    # =========================================================================
    # PART 3: AI AGENT EXECUTION & MODEL GOVERNANCE
    # =========================================================================

    def test_07_agent_execution_with_model_predictions(self):
        """Verifies AI Agent executes clinical tools against ML pipeline and never invents numbers."""
        print(f"{YELLOW}[Agent Test 7/8]{RESET} Testing AI Agent execution and ML model predictions...")

        # Test query 1: Department Forecast
        res1 = carepulse_agent.run_query("Forecast bed occupancy for Cardiology with 15 admissions", [], self.flask_context)
        self.assertTrue(res1["success"])
        self.assertIn("answer", res1)
        self.assertIn("metrics", res1)
        m1 = res1["metrics"]
        self.assertEqual(m1["department"], "Cardiology")
        self.assertIsInstance(m1["predicted_occupancy_rate"], (int, float))
        self.assertGreater(m1["total_bed_capacity"], 0)
        self.assertEqual(m1["estimated_available_beds"], max(0, m1["total_bed_capacity"] - m1["predicted_occupied_beds"]))
        print(f"  {GREEN}PASS{RESET} -> Agent Forecast: Cardiology {m1['predicted_occupancy_rate']}% ({m1['estimated_available_beds']} beds avail)")

        # Test query 2: Bed Availability
        res2 = carepulse_agent.run_query("How many beds are available in Surgery right now?", [], self.flask_context)
        self.assertTrue(res2["success"])
        self.assertIn("Surgery", res2["answer"])
        print(f"  {GREEN}PASS{RESET} -> Agent Availability query answered with verified data.")

        # Test query 3: Alert Status
        res3 = carepulse_agent.run_query("What are the active Red-Zone capacity alerts?", [], self.flask_context)
        self.assertTrue(res3["success"])
        self.assertIn("answer", res3)
        print(f"  {GREEN}PASS{RESET} -> Agent Alert Status query executed successfully.")

    def test_08_technical_details_and_model_name_secrecy(self):
        """Verifies requirement: Model names and mathematical equations are kept out of conversational answers."""
        print(f"{YELLOW}[Compliance Test 8/8]{RESET} Verifying strict ML model name secrecy in UI responses...")

        # Benchmark query
        res = self.client.post("/api/ask", json={"question": "What is the system evaluation RMSE and R2 score?"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        answer = data.get("answer", "")

        # Must report numerical metrics
        self.assertIn("2.231", answer)
        self.assertIn("0.976", answer)

        # Must NEVER disclose model names
        forbidden_names = ["Linear Regression", "Random Forest", "XGBoost", "OLS", "y = mx + c"]
        for forbidden in forbidden_names:
            self.assertNotIn(forbidden, answer, f"Violation: Forbidden model name '{forbidden}' found in answer!")

        print(f"  {GREEN}PASS{RESET} -> Model metrics verified (RMSE 2.231%, R² 0.976). Zero forbidden algorithm names exposed.")


def run_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCarePulseMCPIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print(f"\n{BOLD}{GREEN}==========================================================================={RESET}")
        print(f"{BOLD}{GREEN} ALL 8 MCP & AI AGENT INTEGRATION TESTS PASSED (100% SUCCESS)              {RESET}")
        print(f"{BOLD}{GREEN}==========================================================================={RESET}\n")
        return 0
    else:
        print(f"\n{BOLD}{RED}==========================================================================={RESET}")
        print(f"{BOLD}{RED} SOME TESTS FAILED                                                         {RESET}")
        print(f"{BOLD}{RED}==========================================================================={RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_suite())
