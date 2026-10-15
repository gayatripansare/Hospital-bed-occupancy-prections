"""
=============================================================================
CAREPULSE ANALYTICS - INTELLIGENT BED OCCUPANCY ALERT SYSTEM
Project 14: Hospital Bed Occupancy Forecasting
=============================================================================
Continuous evaluation of hospital bed occupancy rates and zone management:
- Green Zone : Occupancy < 70% (Normal Operations)
- Yellow Zone: Occupancy 70% to < 85% (Elevated Stress / Warning)
- Red Zone   : Occupancy >= 85% (Critical Capacity Alert)

Configurable thresholds, real-time trigger evaluation, forecast-based
early warnings, deduplication, and lifecycle alert management.
=============================================================================
"""

import os
import json
import uuid
import threading
from datetime import datetime
import pandas as pd

ALERT_CONFIG_FILE = "data/alert_config.json"
ALERT_HISTORY_FILE = "data/alerts_history.json"
_ALERT_LOCK = threading.Lock()

# Default thresholds
DEFAULT_THRESHOLDS = {
    "green_max": 70.0,
    "red_min": 85.0
}


def load_thresholds():
    """Loads configurable occupancy zone thresholds from disk or defaults."""
    if os.path.exists(ALERT_CONFIG_FILE):
        try:
            with open(ALERT_CONFIG_FILE, 'r') as f:
                data = json.load(f)
                green = float(data.get("green_max", DEFAULT_THRESHOLDS["green_max"]))
                red = float(data.get("red_min", DEFAULT_THRESHOLDS["red_min"]))
                if 0 < green < red <= 100:
                    return {"green_max": green, "red_min": red}
        except Exception as e:
            print(f"Warning: Could not read alert config: {e}")
    return DEFAULT_THRESHOLDS.copy()


def save_thresholds(green_max, red_min):
    """Saves updated thresholds with strict validation."""
    if not (0 < green_max < red_min <= 100):
        raise ValueError(f"Invalid thresholds: green_max ({green_max}) must be less than red_min ({red_min}) and within 0-100.")
    os.makedirs(os.path.dirname(ALERT_CONFIG_FILE), exist_ok=True)
    payload = {"green_max": round(float(green_max), 1), "red_min": round(float(red_min), 1)}
    with open(ALERT_CONFIG_FILE, 'w') as f:
        json.dump(payload, f, indent=2)
    return payload


def classify_occupancy_zone(rate, thresholds=None):
    """
    Classifies an occupancy rate into Green, Yellow, or Red zone.
    Returns: dict(zone, label, description, color)
    """
    if rate is None or pd.isna(rate):
        return {
            "zone": "UNKNOWN",
            "label": "Unknown / Missing Data",
            "description": "Bed occupancy data is unavailable or unverified.",
            "color": "grey"
        }

    t = thresholds or load_thresholds()
    green_max = t["green_max"]
    red_min = t["red_min"]

    rate_val = float(rate)
    if rate_val < green_max:
        return {
            "zone": "GREEN",
            "label": f"Green Zone (< {green_max:.0f}%)",
            "description": "Normal Occupancy. Inpatient bed capacity buffer is adequate.",
            "color": "green",
            "severity": "normal"
        }
    elif rate_val < red_min:
        return {
            "zone": "YELLOW",
            "label": f"Yellow Zone ({green_max:.0f}% – {red_min:.0f}%)",
            "description": "Warning Status. Elevated bed utilization requiring close monitoring.",
            "color": "yellow",
            "severity": "warning"
        }
    else:
        return {
            "zone": "RED",
            "label": f"Red Zone (≥ {red_min:.0f}%)",
            "description": "Critical Capacity Stress. Emergency capacity assessment recommended.",
            "color": "red",
            "severity": "critical"
        }


def validate_and_calc_bed_capacity(total_beds, occupied_beds):
    """
    Validates hospital bed values and calculates:
      Occupancy Rate = (Occupied Beds / Total Beds) * 100
      Available Beds = Total Beds - Occupied Beds
    Safely rejects invalid, missing, or negative numbers without fabrication.
    """
    if total_beds is None or occupied_beds is None:
        return {
            "valid": False,
            "error": "Bed capacity data is missing.",
            "occupancy_rate": None,
            "available_beds": None,
            "total_beds": None,
            "occupied_beds": None
        }

    try:
        tot = float(total_beds)
        occ = float(occupied_beds)
    except (ValueError, TypeError):
        return {
            "valid": False,
            "error": "Bed numbers must be numeric values.",
            "occupancy_rate": None,
            "available_beds": None,
            "total_beds": None,
            "occupied_beds": None
        }

    if tot <= 0:
        return {
            "valid": False,
            "error": "Total hospital beds must be greater than zero.",
            "occupancy_rate": None,
            "available_beds": None,
            "total_beds": tot,
            "occupied_beds": occ
        }

    if occ < 0:
        return {
            "valid": False,
            "error": "Occupied beds cannot be negative.",
            "occupancy_rate": None,
            "available_beds": None,
            "total_beds": tot,
            "occupied_beds": occ
        }

    if occ > tot:
        return {
            "valid": False,
            "error": f"Occupied beds ({occ}) exceeds total capacity ({tot}).",
            "occupancy_rate": None,
            "available_beds": None,
            "total_beds": tot,
            "occupied_beds": occ
        }

    rate = round((occ / tot) * 100.0, 1)
    avail = int(tot - occ)

    return {
        "valid": True,
        "error": None,
        "occupancy_rate": rate,
        "available_beds": avail,
        "total_beds": int(tot),
        "occupied_beds": int(occ)
    }


# =============================================================================
# ALERT STORE & DEDUPLICATION LOGIC
# =============================================================================

class AlertStore:
    def __init__(self, history_file=ALERT_HISTORY_FILE):
        self.history_file = history_file
        self._alerts = []
        self._load_from_disk()

    def _load_from_disk(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self._alerts = json.load(f)
            except Exception as e:
                print(f"Warning: Could not read alerts history: {e}")
                self._alerts = []

    def _save_to_disk(self):
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self._alerts, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not write alerts history: {e}")

    def get_all_alerts(self):
        return sorted(self._alerts, key=lambda x: x.get("timestamp", ""), reverse=True)

    def get_active_alerts(self):
        return [a for a in self.get_all_alerts() if a.get("status") == "ACTIVE"]

    def find_active_alert(self, department, trigger_type):
        """Finds if an active alert already exists for (department, trigger_type)."""
        for a in self._alerts:
            if a.get("status") == "ACTIVE" and a.get("department") == department and a.get("trigger_type") == trigger_type:
                return a
        return None

    def create_or_update_alert(self, department, trigger_type, occupancy_rate, available_beds,
                               total_beds, occupied_beds, threshold_val, forecast_date=None):
        """
        Deduplication rule:
        If an active alert already exists for this (department, trigger_type),
        update its latest observed values and refresh timestamp without generating a duplicate ID.
        If no active alert exists, create a brand-new alert.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        is_predicted = (trigger_type == "PREDICTED_OCCUPANCY")
        
        rate_val = round(float(occupancy_rate), 1)
        avail_beds = int(available_beds)
        tot_beds = int(total_beds)

        if is_predicted:
            title = "Predicted Red-Zone Warning"
            msg = (
                f"PREDICTED WARNING: ML forecasting indicates hospital occupancy for {department} "
                f"is expected to reach {rate_val}% on {forecast_date or 'the next cycle'}. "
                f"The department will enter the RED ZONE (≥ {threshold_val:.0f}%). "
                f"Estimated available beds remaining: {avail_beds} out of {tot_beds}. "
                f"Early capacity planning and discharge scheduling are recommended."
            )
            explanation = "Triggered by ML predictive forecasting model before acute capacity pressure occurs."
        else:
            title = "Current Red-Zone Alert"
            msg = (
                f"CRITICAL ALERT: Hospital occupancy in {department} has reached {rate_val}%. "
                f"The department is in the RED ZONE (≥ {threshold_val:.0f}%). "
                f"Current available beds: {avail_beds} out of {tot_beds}. "
                f"Immediate capacity assessment and resource planning are recommended."
            )
            explanation = "Triggered by actual verified inpatient census crossing the red-zone threshold."

        existing = self.find_active_alert(department, trigger_type)
        if existing:
            # Update existing active alert to prevent duplicate spam on refresh
            existing["occupancy_rate"] = rate_val
            existing["available_beds"] = avail_beds
            existing["occupied_beds"] = int(occupied_beds)
            existing["total_beds"] = tot_beds
            existing["last_evaluated_at"] = now_str
            existing["message"] = msg
            if forecast_date:
                existing["forecast_date"] = forecast_date
            self._save_to_disk()
            return existing, False

        # Create new alert
        new_alert = {
            "id": f"ALT-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": now_str,
            "last_evaluated_at": now_str,
            "department": department,
            "trigger_type": trigger_type,  # CURRENT_OCCUPANCY or PREDICTED_OCCUPANCY
            "title": title,
            "severity": "CRITICAL",
            "zone": "RED",
            "occupancy_rate": rate_val,
            "available_beds": avail_beds,
            "occupied_beds": int(occupied_beds),
            "total_beds": tot_beds,
            "threshold": threshold_val,
            "forecast_date": forecast_date,
            "message": msg,
            "explanation": explanation,
            "status": "ACTIVE",  # ACTIVE or ACKNOWLEDGED
            "acknowledged_at": None,
            "acknowledged_by": None
        }

        self._alerts.append(new_alert)
        self._save_to_disk()
        return new_alert, True

    def acknowledge_alert(self, alert_id, user="Hospital Staff"):
        """Marks an alert as acknowledged by an authorized user."""
        for a in self._alerts:
            if a.get("id") == alert_id:
                a["status"] = "ACKNOWLEDGED"
                a["acknowledged_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                a["acknowledged_by"] = user
                self._save_to_disk()
                return a
        return None

    def auto_resolve_if_recovered(self, department, trigger_type, current_rate, red_threshold):
        """
        If a department's occupancy rate drops below the red threshold,
        resolve any active alert for that department.
        """
        if current_rate < red_threshold:
            for a in self._alerts:
                if a.get("status") == "ACTIVE" and a.get("department") == department and a.get("trigger_type") == trigger_type:
                    a["status"] = "RESOLVED"
                    a["resolved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self._save_to_disk()


# Singleton alert store
alert_store = AlertStore()


# =============================================================================
# CONTINUOUS HOSPITAL OCCUPANCY EVALUATOR
# =============================================================================

def evaluate_hospital_alerts(df_raw, execute_forecast_fn=None):
    """
    Evaluates both actual hospital census and predictive ML forecasts
    across all clinical departments and generates Red-Zone alerts.
    """
    with _ALERT_LOCK:
        thresholds = load_thresholds()
        red_threshold = thresholds["red_min"]
        
        if df_raw.empty:
            return {
                "success": False,
                "error": "Dataset is empty; cannot evaluate occupancy."
            }

        latest_date = df_raw['Date'].max()
        latest_df = df_raw[df_raw['Date'] == latest_date].copy()

        departments_summary = []
        active_red_alerts = []

        # 1. Evaluate Current Occupancy per Department
        for _, row in latest_df.iterrows():
            dept = row['Department']
            tot = row.get('Total_Hospital_Beds')
            occ = row.get('Beds_Occupied')

            val_res = validate_and_calc_bed_capacity(tot, occ)
            if not val_res["valid"]:
                departments_summary.append({
                    "department": dept,
                    "date": str(latest_date.date()),
                    "valid": False,
                    "error": val_res["error"],
                    "zone_info": classify_occupancy_zone(None, thresholds)
                })
                continue

            rate = val_res["occupancy_rate"]
            avail = val_res["available_beds"]
            tot_beds = val_res["total_beds"]
            occ_beds = val_res["occupied_beds"]
            zone_info = classify_occupancy_zone(rate, thresholds)

            # Check if actual occupancy reaches RED ZONE
            if rate >= red_threshold:
                alert, created = alert_store.create_or_update_alert(
                    department=dept,
                    trigger_type="CURRENT_OCCUPANCY",
                    occupancy_rate=rate,
                    available_beds=avail,
                    total_beds=tot_beds,
                    occupied_beds=occ_beds,
                    threshold_val=red_threshold
                )
                active_red_alerts.append(alert)
            else:
                alert_store.auto_resolve_if_recovered(dept, "CURRENT_OCCUPANCY", rate, red_threshold)

            # 2. Evaluate Predictive Forecast (if forecasting function provided)
            forecast_info = None
            if execute_forecast_fn:
                try:
                    f_res = execute_forecast_fn(dept)
                    if f_res:
                        pred_rate = f_res.get("predicted_occupancy_rate")
                        pred_avail = f_res.get("estimated_available_beds")
                        pred_occ = f_res.get("predicted_occupied_beds")
                        f_zone = classify_occupancy_zone(pred_rate, thresholds)

                        forecast_info = {
                            "predicted_occupancy_rate": pred_rate,
                            "estimated_available_beds": pred_avail,
                            "predicted_occupied_beds": pred_occ,
                            "zone": f_zone["zone"],
                            "is_peak": f_res.get("is_peak_demand", False)
                        }

                        # Check if forecasted occupancy reaches RED ZONE
                        if pred_rate >= red_threshold:
                            next_date = (latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                            pred_alert, p_created = alert_store.create_or_update_alert(
                                department=dept,
                                trigger_type="PREDICTED_OCCUPANCY",
                                occupancy_rate=pred_rate,
                                available_beds=pred_avail,
                                total_beds=tot_beds,
                                occupied_beds=pred_occ,
                                threshold_val=red_threshold,
                                forecast_date=next_date
                            )
                            active_red_alerts.append(pred_alert)
                        else:
                            alert_store.auto_resolve_if_recovered(dept, "PREDICTED_OCCUPANCY", pred_rate, red_threshold)
                except Exception as ex:
                    print(f"Warning: Forecasting evaluation error for {dept}: {ex}")

            departments_summary.append({
                "department": dept,
                "date": str(latest_date.date()),
                "valid": True,
                "occupancy_rate": rate,
                "available_beds": avail,
                "total_beds": tot_beds,
                "occupied_beds": occ_beds,
                "zone_info": zone_info,
                "forecast_info": forecast_info
            })

        # Hospital-Wide Rollup
        tot_hospital_beds = int(latest_df['Total_Hospital_Beds'].sum())
        tot_hospital_occ = int(latest_df['Beds_Occupied'].sum())
        h_val = validate_and_calc_bed_capacity(tot_hospital_beds, tot_hospital_occ)
        hospital_rate = h_val["occupancy_rate"] if h_val["valid"] else 0.0
        hospital_avail = h_val["available_beds"] if h_val["valid"] else 0
        hospital_zone = classify_occupancy_zone(hospital_rate, thresholds)

        if hospital_rate >= red_threshold:
            h_alert, _ = alert_store.create_or_update_alert(
                department="Hospital-Wide (All Units)",
                trigger_type="CURRENT_OCCUPANCY",
                occupancy_rate=hospital_rate,
                available_beds=hospital_avail,
                total_beds=tot_hospital_beds,
                occupied_beds=tot_hospital_occ,
                threshold_val=red_threshold
            )
            active_red_alerts.append(h_alert)
        else:
            alert_store.auto_resolve_if_recovered("Hospital-Wide (All Units)", "CURRENT_OCCUPANCY", hospital_rate, red_threshold)

        return {
            "success": True,
            "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "latest_census_date": str(latest_date.date()),
            "thresholds": thresholds,
            "hospital_summary": {
                "total_beds": tot_hospital_beds,
                "occupied_beds": tot_hospital_occ,
                "available_beds": hospital_avail,
                "occupancy_rate": hospital_rate,
                "zone_info": hospital_zone
            },
            "departments": departments_summary,
            "active_alerts": alert_store.get_active_alerts(),
            "total_active_alerts": len(alert_store.get_active_alerts())
        }
