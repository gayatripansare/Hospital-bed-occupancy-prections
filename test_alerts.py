"""
=============================================================================
CAREPULSE ANALYTICS - HOSPITAL BED OCCUPANCY ALERT SYSTEM TEST SUITE
Project 14: Hospital Bed Occupancy Forecasting
=============================================================================
Comprehensive test suite verifying:
1. 65% occupancy -> Green Zone (< 70%)
2. 75% occupancy -> Yellow Zone (70% to < 85%)
3. 85% occupancy -> Red Zone (>= 85%)
4. 95% occupancy -> Red Zone (>= 85%)
5. Missing or invalid bed-capacity data handling (None, <=0, negative, > total)
6. Forecasted occupancy crossing red-zone threshold (Early-Warning Alert)
7. Deduplication on repeated evaluations / refreshes (No duplicate active alerts)
8. Alert acknowledgment lifecycle (ACTIVE -> ACKNOWLEDGED)
9. Configurable threshold updates and boundary validation
10. REST API Endpoints (/api/alerts, /api/alerts/thresholds, /api/alerts/acknowledge, /api/occupancy-status)
11. Conversational Assistant natural language query integration for alerts
=============================================================================
"""

import sys
import json
import unittest
from datetime import datetime
import pandas as pd

import alert_system
from app import app, df_raw, execute_department_forecast

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


class TestHospitalBedOccupancyAlertSystem(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        # Ensure default thresholds for consistent testing
        alert_system.save_thresholds(70.0, 85.0)

    # -------------------------------------------------------------------------
    # TEST 1 & 2 & 3 & 4: OCCUPANCY ZONE CLASSIFICATION
    # -------------------------------------------------------------------------
    def test_01_green_zone_65_pct(self):
        """Synthetic test: 65% occupancy must be classified as GREEN ZONE."""
        res = alert_system.classify_occupancy_zone(65.0)
        self.assertEqual(res["zone"], "GREEN")
        self.assertEqual(res["severity"], "normal")
        self.assertIn("< 70%", res["label"])

    def test_02_yellow_zone_75_pct(self):
        """Synthetic test: 75% occupancy must be classified as YELLOW ZONE."""
        res = alert_system.classify_occupancy_zone(75.0)
        self.assertEqual(res["zone"], "YELLOW")
        self.assertEqual(res["severity"], "warning")
        self.assertIn("70% – 85%", res["label"])

    def test_03_red_zone_85_pct(self):
        """Synthetic test: 85% occupancy must be classified as RED ZONE (Threshold boundary)."""
        res = alert_system.classify_occupancy_zone(85.0)
        self.assertEqual(res["zone"], "RED")
        self.assertEqual(res["severity"], "critical")
        self.assertIn("≥ 85%", res["label"])

    def test_04_red_zone_95_pct(self):
        """Synthetic test: 95% occupancy must be classified as RED ZONE."""
        res = alert_system.classify_occupancy_zone(95.0)
        self.assertEqual(res["zone"], "RED")
        self.assertEqual(res["severity"], "critical")
        self.assertIn("≥ 85%", res["label"])

    # -------------------------------------------------------------------------
    # TEST 5: MISSING AND INVALID BED-CAPACITY DATA
    # -------------------------------------------------------------------------
    def test_05_missing_and_invalid_bed_capacity(self):
        """Validates bed calculations safely catch negative, zero, overflow, and None without fabrication."""
        # Valid baseline
        valid_res = alert_system.validate_and_calc_bed_capacity(100, 80)
        self.assertTrue(valid_res["valid"])
        self.assertEqual(valid_res["occupancy_rate"], 80.0)
        self.assertEqual(valid_res["available_beds"], 20)

        # Missing (None)
        missing_res = alert_system.validate_and_calc_bed_capacity(None, 40)
        self.assertFalse(missing_res["valid"])
        self.assertIn("missing", missing_res["error"].lower())

        # Zero total beds
        zero_res = alert_system.validate_and_calc_bed_capacity(0, 10)
        self.assertFalse(zero_res["valid"])
        self.assertIn("greater than zero", zero_res["error"].lower())

        # Negative occupied beds
        neg_res = alert_system.validate_and_calc_bed_capacity(100, -5)
        self.assertFalse(neg_res["valid"])
        self.assertIn("negative", neg_res["error"].lower())

        # Occupied exceeds total capacity
        over_res = alert_system.validate_and_calc_bed_capacity(50, 60)
        self.assertFalse(over_res["valid"])
        self.assertIn("exceeds total capacity", over_res["error"].lower())

        # Unknown zone for invalid rate
        unk_zone = alert_system.classify_occupancy_zone(None)
        self.assertEqual(unk_zone["zone"], "UNKNOWN")

    # -------------------------------------------------------------------------
    # TEST 6: FORECAST-BASED EARLY WARNING (PREDICTED RED ZONE)
    # -------------------------------------------------------------------------
    def test_06_forecast_based_early_warning(self):
        """Checks that forecast-based early warning generates PREDICTED_OCCUPANCY distinct from CURRENT_OCCUPANCY."""
        temp_store = alert_system.AlertStore("data/test_alerts.json")
        alert, created = temp_store.create_or_update_alert(
            department="Cardiology",
            trigger_type="PREDICTED_OCCUPANCY",
            occupancy_rate=92.5,
            available_beds=4,
            total_beds=60,
            occupied_beds=56,
            threshold_val=85.0,
            forecast_date="2025-01-02"
        )
        self.assertEqual(alert["trigger_type"], "PREDICTED_OCCUPANCY")
        self.assertEqual(alert["title"], "Predicted Red-Zone Warning")
        self.assertIn("PREDICTED WARNING", alert["message"])
        self.assertIn("2025-01-02", alert["message"])
        self.assertEqual(alert["status"], "ACTIVE")

        # Verify distinct from current occupancy alert
        cur_alert, cur_created = temp_store.create_or_update_alert(
            department="Cardiology",
            trigger_type="CURRENT_OCCUPANCY",
            occupancy_rate=88.0,
            available_beds=7,
            total_beds=60,
            occupied_beds=53,
            threshold_val=85.0
        )
        self.assertEqual(cur_alert["trigger_type"], "CURRENT_OCCUPANCY")
        self.assertEqual(cur_alert["title"], "Current Red-Zone Alert")
        self.assertIn("CRITICAL ALERT", cur_alert["message"])
        self.assertNotEqual(alert["id"], cur_alert["id"])

        # Clean up test file
        import os
        if os.path.exists("data/test_alerts.json"):
            os.remove("data/test_alerts.json")

    # -------------------------------------------------------------------------
    # TEST 7: DEDUPLICATION ON REPEATED EVALUATIONS
    # -------------------------------------------------------------------------
    def test_07_repeated_evaluations_without_duplicates(self):
        """Guarantees that repeated refreshes and evaluations update existing active alerts without creating duplicates."""
        # Initial evaluation
        res1 = alert_system.evaluate_hospital_alerts(df_raw, execute_department_forecast)
        count1 = len(alert_system.alert_store.get_active_alerts())
        self.assertGreater(count1, 0)

        # Second immediate evaluation
        res2 = alert_system.evaluate_hospital_alerts(df_raw, execute_department_forecast)
        count2 = len(alert_system.alert_store.get_active_alerts())
        self.assertEqual(count1, count2, "Duplicate alerts detected on re-evaluation!")

        # Third immediate evaluation
        res3 = alert_system.evaluate_hospital_alerts(df_raw, execute_department_forecast)
        count3 = len(alert_system.alert_store.get_active_alerts())
        self.assertEqual(count1, count3, "Duplicate alerts detected on third re-evaluation!")

    # -------------------------------------------------------------------------
    # TEST 8: ALERT ACKNOWLEDGMENT LIFECYCLE
    # -------------------------------------------------------------------------
    def test_08_alert_acknowledgment(self):
        """Verifies that hospital staff can acknowledge active alerts."""
        active = alert_system.alert_store.get_active_alerts()
        self.assertTrue(len(active) > 0, "Expected at least one active alert for test")
        target_id = active[0]["id"]

        # Acknowledge via AlertStore
        ack = alert_system.alert_store.acknowledge_alert(target_id, user="Dr. Adams (Chief of Medicine)")
        self.assertIsNotNone(ack)
        self.assertEqual(ack["status"], "ACKNOWLEDGED")
        self.assertEqual(ack["acknowledged_by"], "Dr. Adams (Chief of Medicine)")
        self.assertIsNotNone(ack["acknowledged_at"])

        # Check via API endpoint
        active_remaining = alert_system.alert_store.get_active_alerts()
        if active_remaining:
            next_id = active_remaining[0]["id"]
            resp = self.client.post('/api/alerts/acknowledge', json={
                "alert_id": next_id,
                "user": "Nurse Supervisor"
            })
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data["success"])
            self.assertEqual(data["alert"]["status"], "ACKNOWLEDGED")

    # -------------------------------------------------------------------------
    # TEST 9: CONFIGURABLE THRESHOLD UPDATES AND VALIDATION
    # -------------------------------------------------------------------------
    def test_09_configurable_thresholds(self):
        """Verifies configurable thresholds update and invalid boundaries are rejected."""
        # Valid update
        resp = self.client.post('/api/alerts/thresholds', json={
            "green_max": 65.0,
            "red_min": 80.0
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["thresholds"]["green_max"], 65.0)
        self.assertEqual(resp.get_json()["thresholds"]["red_min"], 80.0)

        # Invalid: green_max >= red_min
        bad_resp = self.client.post('/api/alerts/thresholds', json={
            "green_max": 85.0,
            "red_min": 80.0
        })
        self.assertEqual(bad_resp.status_code, 400)
        self.assertFalse(bad_resp.get_json()["success"])

        # Invalid: out of 0-100 range
        out_resp = self.client.post('/api/alerts/thresholds', json={
            "green_max": -10.0,
            "red_min": 120.0
        })
        self.assertEqual(out_resp.status_code, 400)

        # Restore defaults
        alert_system.save_thresholds(70.0, 85.0)

    # -------------------------------------------------------------------------
    # TEST 10: FLASK REST API ENDPOINTS
    # -------------------------------------------------------------------------
    def test_10_api_endpoints_integrity(self):
        """Verifies all Alert System REST API endpoints return valid JSON and status 200."""
        # GET /api/alerts
        res_alerts = self.client.get('/api/alerts')
        self.assertEqual(res_alerts.status_code, 200)
        d_alerts = res_alerts.get_json()
        self.assertTrue(d_alerts["success"])
        self.assertIn("active_alerts", d_alerts)
        self.assertIn("thresholds", d_alerts)
        self.assertIn("hospital_summary", d_alerts)

        # GET /api/alerts/thresholds
        res_thresh = self.client.get('/api/alerts/thresholds')
        self.assertEqual(res_thresh.status_code, 200)
        self.assertEqual(res_thresh.get_json()["thresholds"]["red_min"], 85.0)

        # GET /api/occupancy-status
        res_status = self.client.get('/api/occupancy-status')
        self.assertEqual(res_status.status_code, 200)
        d_status = res_status.get_json()
        self.assertTrue(d_status["success"])
        self.assertIn("status", d_status)

        # POST /api/alerts/evaluate
        res_eval = self.client.post('/api/alerts/evaluate')
        self.assertEqual(res_eval.status_code, 200)
        self.assertTrue(res_eval.get_json()["success"])

    # -------------------------------------------------------------------------
    # TEST 11: CONVERSATIONAL ASSISTANT ALERT INTEGRATION
    # -------------------------------------------------------------------------
    def test_11_conversational_alert_queries(self):
        """Checks that conversational assistant answers natural language questions about alerts."""
        resp = self.client.post('/api/ask', json={
            "question": "Are there any active alerts or departments in the red zone?"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("Bed Occupancy Alert Status", data["answer"])
        self.assertIn("Red Zone", data["answer"])
        self.assertIsNotNone(data["metrics"])


if __name__ == '__main__':
    print(f"\n{BOLD}{CYAN}==========================================================================={RESET}")
    print(f"{BOLD}{CYAN}   CAREPULSE ANALYTICS - BED OCCUPANCY ALERT SYSTEM VERIFICATION SUITE       {RESET}")
    print(f"{BOLD}{CYAN}==========================================================================={RESET}\n")

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestHospitalBedOccupancyAlertSystem)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print(f"\n{BOLD}{GREEN}ALL {result.testsRun} ALERT SYSTEM TESTS PASSED SUCCESSFULLY (100% SUCCESS){RESET}\n")
        sys.exit(0)
    else:
        print(f"\n{BOLD}{RED}SOME TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors{RESET}\n")
        sys.exit(1)
