"""
=============================================================================
CAREPULSE ANALYTICS - CONVERSATIONAL RESEARCH ASSISTANT BACKEND
Project 14: Hospital Bed Occupancy Forecasting
=============================================================================
A conversational research-assistant interface for hospital bed occupancy,
bed availability, and peak demand forecasting.
- Local regression model: Linear Regression (RMSE: 2.231%, R²: 0.976)
- Dataset: data/hospital_occupancy.csv (8,768 records, 8 clinical departments)
- No external LLM, paid APIs, or synthetic hallucinations
=============================================================================
"""

import os
import re
import json
import joblib
import threading
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, Response
import charts
import alert_system
from ai_agent import carepulse_agent

app = Flask(__name__)

DATA_PATH = "data/hospital_occupancy.csv"
MODEL_DIR = "model"

# ---------- Graceful Artifact Loading ----------
best_model = None
preprocessor = None
metadata = {}
candidate_models = {}

try:
    if os.path.exists(os.path.join(MODEL_DIR, "best_model.pkl")):
        best_model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    if os.path.exists(os.path.join(MODEL_DIR, "preprocessor.pkl")):
        preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    if os.path.exists(os.path.join(MODEL_DIR, "model_metadata.json")):
        with open(os.path.join(MODEL_DIR, "model_metadata.json"), 'r') as f:
            metadata = json.load(f)
            
    for m_key in ['linear_regression', 'random_forest', 'xgboost']:
        m_file = os.path.join(MODEL_DIR, f"{m_key}.pkl")
        if os.path.exists(m_file):
            candidate_models[m_key] = joblib.load(m_file)
except Exception as e:
    print(f"Warning: Artifact loading encountered an issue: {e}")

# ---------- Load Historical Dataset ----------
try:
    df_raw = pd.read_csv(DATA_PATH)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_raw = df_raw.sort_values(['Department', 'Date']).reset_index(drop=True)
    # Warm up charts cache in a background thread for zero-latency chart delivery
    threading.Thread(target=charts.warm_up_cache, daemon=True).start()
except Exception as e:
    df_raw = pd.DataFrame()
    print(f"Warning: Could not read dataset at {DATA_PATH}: {e}")

# Department Aliases
DEPT_ALIASES = {
    'cardiology': 'Cardiology', 'cardio': 'Cardiology', 'heart': 'Cardiology',
    'general medicine': 'General Medicine', 'general med': 'General Medicine', 'gen med': 'General Medicine', 'internal medicine': 'General Medicine',
    'neurology': 'Neurology', 'neuro': 'Neurology', 'brain': 'Neurology',
    'obstetrics & gynecology': 'Obstetrics & Gynecology', 'obstetrics and gynecology': 'Obstetrics & Gynecology',
    'obstetrics': 'Obstetrics & Gynecology', 'gynecology': 'Obstetrics & Gynecology', 'obgyn': 'Obstetrics & Gynecology', 'maternity': 'Obstetrics & Gynecology',
    'oncology': 'Oncology', 'onco': 'Oncology', 'cancer': 'Oncology',
    'orthopedics': 'Orthopedics', 'ortho': 'Orthopedics', 'bone': 'Orthopedics',
    'pediatrics': 'Pediatrics', 'peds': 'Pediatrics', 'children': 'Pediatrics', 'child': 'Pediatrics',
    'surgery': 'Surgery', 'surgical': 'Surgery', 'surg': 'Surgery', 'operation': 'Surgery', 'or': 'Surgery'
}

MONTH_MAP = {
    'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
    'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
    'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
    'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
}


# =============================================================================
# CORE ML FORECASTING ENGINE
# =============================================================================

def execute_department_forecast(dept_name, custom_adm=None, custom_emerg=None, custom_disc=None, season="Winter", month=1, is_holiday=0):
    """Executes regression inference for a specific department and parameters."""
    if best_model is None or preprocessor is None or df_raw.empty:
        return None

    dept_rows = df_raw[df_raw['Department'] == dept_name]
    if dept_rows.empty:
        return None

    last_row = dept_rows.sort_values('Date').iloc[-1]
    last_7 = dept_rows.sort_values('Date').iloc[-7:]
    
    tot_beds = int(last_row['Total_Hospital_Beds'])
    hist_occ = float(last_row['Occupancy_Rate_Pct'])
    hist_adm = int(last_row['Daily_Admissions'])
    hist_emerg = int(last_row['Emergency_Admissions'])
    hist_disc = int(last_row['Daily_Discharges'])
    hist_los = float(last_row['Avg_Length_of_Stay'])
    hist_staff = float(last_row['Staff_Availability_Pct'])

    adm = float(custom_adm) if custom_adm is not None else float(hist_adm)
    emerg = float(custom_emerg) if custom_emerg is not None else float(hist_emerg)
    disc = float(custom_disc) if custom_disc is not None else float(hist_disc)

    feat_dict = {
        'Total_Hospital_Beds': [tot_beds],
        'Month': [month],
        'Public_Holiday': [is_holiday],
        'Special_Event': [0],
        'Occupancy_Lag_1': [hist_occ],
        'Occupancy_Lag_2': [dept_rows.sort_values('Date').iloc[-2]['Occupancy_Rate_Pct']],
        'Occupancy_Lag_7': [dept_rows.sort_values('Date').iloc[-7]['Occupancy_Rate_Pct']],
        'Occupancy_Roll_Mean_7': [float(last_7['Occupancy_Rate_Pct'].mean())],
        'Admissions_Lag_1': [adm],
        'Emergency_Lag_1': [emerg],
        'Discharges_Lag_1': [disc],
        'Avg_LOS_Lag_1': [hist_los],
        'Staff_Avail_Lag_1': [hist_staff],
        'Department': [dept_name],
        'Season': [season],
        'Day_of_Week': ['Monday']
    }

    feat_df = pd.DataFrame(feat_dict)
    X = preprocessor.transform(feat_df)
    pred_rate = float(best_model.predict(X)[0])
    pred_rate = round(min(100.0, max(0.0, pred_rate)), 1)

    pred_occ = int(round((pred_rate * tot_beds) / 100.0))
    pred_avail = max(0, tot_beds - pred_occ)
    is_peak = bool(pred_rate >= 85.0)

    if pred_rate >= 85.0:
        demand_status = "Peak Demand (High Capacity Stress)"
    elif pred_rate >= 75.0:
        demand_status = "Moderate Demand (Normal Operations)"
    else:
        demand_status = "Optimal Demand (Adequate Capacity Buffer)"

    return {
        "department": dept_name,
        "total_bed_capacity": tot_beds,
        "predicted_occupancy_rate": pred_rate,
        "predicted_occupied_beds": pred_occ,
        "estimated_available_beds": pred_avail,
        "is_peak_demand": is_peak,
        "demand_level": demand_status,
        "inputs": {
            "admissions": adm,
            "emergencies": emerg,
            "discharges": disc,
            "season": season
        }
    }


def execute_7day_horizon(dept_name):
    """Executes 7-day forward autoregressive horizon for a department."""
    if best_model is None or preprocessor is None or df_raw.empty:
        return []

    dept_rows = df_raw[df_raw['Department'] == dept_name].sort_values('Date').copy()
    if dept_rows.empty:
        return []

    latest_date = df_raw['Date'].max()
    tot_beds = int(dept_rows.iloc[-1]['Total_Hospital_Beds'])
    recent_rates = dept_rows['Occupancy_Rate_Pct'].tail(7).tolist()
    last_adm = dept_rows['Daily_Admissions'].iloc[-1]
    last_emerg = dept_rows['Emergency_Admissions'].iloc[-1]
    last_disc = dept_rows['Daily_Discharges'].iloc[-1]
    last_los = dept_rows['Avg_Length_of_Stay'].iloc[-1]
    last_staff = dept_rows['Staff_Availability_Pct'].iloc[-1]

    days = []
    for step in range(1, 8):
        target_date = latest_date + pd.Timedelta(days=step)
        month = target_date.month
        season = 'Winter' if month in [12, 1, 2] else ('Spring' if month in [3, 4, 5] else ('Summer' if month in [6, 7, 8] else 'Fall'))
        is_holiday = 1 if (month == 1 and target_date.day == 1) else 0

        feat_dict = {
            'Total_Hospital_Beds': [tot_beds],
            'Month': [month],
            'Public_Holiday': [is_holiday],
            'Special_Event': [0],
            'Occupancy_Lag_1': [recent_rates[-1]],
            'Occupancy_Lag_2': [recent_rates[-2]],
            'Occupancy_Lag_7': [recent_rates[0]],
            'Occupancy_Roll_Mean_7': [float(np.mean(recent_rates[-7:]))],
            'Admissions_Lag_1': [last_adm],
            'Emergency_Lag_1': [last_emerg],
            'Discharges_Lag_1': [last_disc],
            'Avg_LOS_Lag_1': [last_los],
            'Staff_Avail_Lag_1': [last_staff],
            'Department': [dept_name],
            'Season': [season],
            'Day_of_Week': [target_date.day_name()]
        }

        X = preprocessor.transform(pd.DataFrame(feat_dict))
        rate = round(min(100.0, max(0.0, float(best_model.predict(X)[0]))), 1)
        occ = int(round((rate * tot_beds) / 100.0))
        avail = max(0, tot_beds - occ)
        recent_rates.append(rate)

        days.append({
            "date": str(target_date.date()),
            "day": target_date.day_name()[:3],
            "rate": rate,
            "occupied": occ,
            "available": avail,
            "is_peak": bool(rate >= 85.0)
        })

    return days


# =============================================================================
# CONVERSATIONAL INTENT RESOLVER & ASSISTANT LOGIC
# =============================================================================

def parse_conversational_query(text, history=[]):
    """
    Extracts clinical entities, intent, and conversational context
    from natural language questions.
    """
    q = text.lower()

    # 1. Extract Department (Current Turn or Inherited from History)
    dept = None
    for alias, canonical in DEPT_ALIASES.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', q):
            dept = canonical
            break

    # Contextual inheritance for follow-up questions
    if dept is None and history:
        for prev in reversed(history):
            prev_content = prev.get('content', '').lower()
            for alias, canonical in DEPT_ALIASES.items():
                if re.search(r'\b' + re.escape(alias) + r'\b', prev_content):
                    dept = canonical
                    break
            if dept:
                break

    # 2. Extract Numbers for What-If Scenarios
    adm = None
    m_adm = re.search(r'(\d+)\s*(?:daily\s*)?admissions?', q)
    if not m_adm:
        m_adm = re.search(r'admissions?\s*(?:of\s*|to\s*|=\s*|is\s*)?(\d+)', q)
    if m_adm:
        adm = float(m_adm.group(1))

    emerg = None
    m_emerg = re.search(r'(\d+)\s*emergenc(?:y|ies)', q)
    if not m_emerg:
        m_emerg = re.search(r'emergenc(?:y|ies)\s*(?:of\s*|to\s*|=\s*|is\s*)?(\d+)', q)
    if m_emerg:
        emerg = float(m_emerg.group(1))

    disc = None
    m_disc = re.search(r'(\d+)\s*discharges?', q)
    if m_disc:
        disc = float(m_disc.group(1))

    # 3. Detect Season & Month
    season = "Winter"
    for s in ['spring', 'summer', 'fall', 'autumn', 'winter']:
        if s in q:
            season = 'Fall' if s == 'autumn' else s.capitalize()
            break

    month = 1
    for m_name, m_num in MONTH_MAP.items():
        if re.search(r'\b' + m_name + r'\b', q):
            month = m_num
            break

    is_holiday = 1 if ('holiday' in q or 'public holiday' in q) else 0

    return {
        "department": dept,
        "admissions": adm,
        "emergencies": emerg,
        "discharges": disc,
        "season": season,
        "month": month,
        "is_holiday": is_holiday,
        "raw_text": q
    }


def generate_research_answer(query_info):
    """
    Synthesizes the ML model output and clinical dataset facts
    into a professional research-assistant response.
    """
    q = query_info['raw_text']
    dept = query_info['department']
    adm = query_info['admissions']
    emerg = query_info['emergencies']
    disc = query_info['discharges']

    # -------------------------------------------------------------
    # INTENT: Model Evaluation, Accuracy & Benchmark Metrics
    # -------------------------------------------------------------
    if any(k in q for k in ['rmse', 'mae', 'r2', 'r^2', 'r²', 'metric', 'accuracy', 'evaluation', 'benchmark', 'how accurate', 'error score']):
        resp = f"""### Forecasting System Evaluation & Benchmark Report

The hospital forecasting model was rigorously benchmarked on the **chronological held-out 2024 test split (2,928 samples)** with zero future data leakage:

* **Primary Production Forecasting Model:**
  - **RMSE:** 2.231% (Primary Root Mean Squared Error)
  - **MAE:** 1.828% (Mean Absolute Error in bed occupancy rate)
  - **R² Score:** 0.976 (Explains 97.6% of occupancy variance)
* **Alternative Candidate Model A:** RMSE: 2.762% | MAE: 2.062% | R²: 0.963
* **Alternative Candidate Model B:** RMSE: 2.947% | MAE: 1.938% | R²: 0.957
* **Historical Persistence Baseline:** RMSE: 7.366% | MAE: 5.206% | R²: 0.734

**Key Finding:** The validated production forecasting model cuts prediction error by **69.7%** compared to the naive persistence baseline (yesterday's occupancy)."""

        return {
            "answer": resp,
            "metrics": None,
            "suggestions": [
                "Forecast bed occupancy for Cardiology tomorrow",
                "How many beds are available in Surgery?",
                "What dataset limitations exist?"
            ]
        }

    # -------------------------------------------------------------
    # INTENT: Dataset Limitations & PDF Feature Gaps
    # -------------------------------------------------------------
    if any(k in q for k in ['limitation', 'gap', 'icd', 'patient type', 'disease category', 'dataset source', 'synthetic']):
        resp = f"""### Project 14 Dataset Specifications & Clinical Limitations

* **Dataset:** <code>data/hospital_occupancy.csv</code> (8,768 records, 8 clinical departments across 1,096 consecutive days, 2022–2024).
* **Documented Clinical Gaps (PDF Project 14):**
  - **Patient Type:** The dataset does not provide individual patient demographics (e.g., inpatient vs outpatient categories).
  - **Disease Category:** Specific ICD-10 diagnostic codes are absent in the dataset.
* **Integrity Commitment:** Rather than fabricating synthetic disease labels, the forecasting model relies on verified operational attributes: Department specialties, astronomical Season, Daily Admissions, Emergency Admissions, and Average Length of Stay."""

        return {
            "answer": resp,
            "metrics": None,
            "suggestions": [
                "What is the system RMSE and R² score?",
                "Forecast General Medicine bed occupancy",
                "Show latest observed hospital census"
            ]
        }

    # -------------------------------------------------------------
    # INTENT: 7-Day Horizon Outlook
    # -------------------------------------------------------------
    if any(k in q for k in ['7 day', '7-day', 'next week', 'horizon', 'weekly outlook', 'upcoming week']) and dept:
        horizon = execute_7day_horizon(dept)
        if horizon:
            rows = ""
            for h in horizon:
                status = "**PEAK DEMAND**" if h['is_peak'] else "Optimal"
                rows += f"- **{h['day']} ({h['date']}):** Projected Rate: **{h['rate']}%** | Occupied: **{h['occupied']} beds** | Available: **{h['available']} beds** | Status: {status}\n"

            resp = f"""### 7-Day Forward Horizon Outlook: {dept}

Here is the autoregressive multi-step bed occupancy forecast for the upcoming 7 days:

{rows}
**Operational Context:** Total departmental capacity is **{horizon[0]['occupied'] + horizon[0]['available']} beds**. Peak demand alert triggers whenever projected occupancy reaches or exceeds **85.0%**."""

            return {
                "answer": resp,
                "metrics": {
                    "department": dept,
                    "horizon_days": 7
                },
                "suggestions": [
                    f"What if emergency admissions surge in {dept}?",
                    "What about Surgery?",
                    "What is the latest observed status?"
                ]
            }

    # -------------------------------------------------------------
    # INTENT: Latest Observed Census (December 31, 2024)
    # -------------------------------------------------------------
    if any(k in q for k in ['latest', 'current', 'census', 'hospital wide', 'hospital-wide', 'how many beds does the hospital have', 'december 31', 'observed']):
        if not df_raw.empty:
            latest_date = df_raw['Date'].max()
            latest_df = df_raw[df_raw['Date'] == latest_date]
            tot_beds = int(latest_df['Total_Hospital_Beds'].sum())
            tot_occ = int(latest_df['Beds_Occupied'].sum())
            tot_avail = tot_beds - tot_occ
            rate = round((tot_occ / tot_beds) * 100.0, 1)

            dept_summary = ""
            for _, r in latest_df.iterrows():
                is_p = r['Occupancy_Rate_Pct'] >= 85.0
                tag = "[PEAK DEMAND]" if is_p else "[Normal]"
                dept_summary += f"- **{r['Department']}:** {r['Beds_Occupied']}/{r['Total_Hospital_Beds']} beds occupied (**{r['Occupancy_Rate_Pct']}%**) | **{r['Total_Hospital_Beds'] - r['Beds_Occupied']} available** {tag}\n"

            resp = f"""### Latest Observed Hospital Census (Date: {latest_date.date()})

Across all 8 clinical departments, here is the observed inpatient census:

* **Total Hospital Bed Capacity:** **{tot_beds} beds**
* **Total Occupied Beds:** **{tot_occ} beds**
* **Total Estimated Available Beds:** **{tot_avail} beds**
* **Overall Hospital Occupancy Rate:** **{rate}%**

#### Department Breakdown:
{dept_summary}
*Note: This represents actual historical observations recorded in the dataset on the final observation date.*"""

            return {
                "answer": resp,
                "metrics": {
                    "total_hospital_beds": tot_beds,
                    "total_beds_occupied": tot_occ,
                    "total_beds_available": tot_avail,
                    "overall_occupancy_rate": rate
                },
                "suggestions": [
                    "Forecast bed occupancy for Cardiology tomorrow",
                    "Will Surgery face peak demand with 15 admissions?",
                    "What is the model evaluation RMSE?"
                ]
            }

    # -------------------------------------------------------------
    # INTENT: Active Alerts & Occupancy Zone Status
    # -------------------------------------------------------------
    if any(k in q for k in ['alert', 'red zone', 'yellow zone', 'green zone', 'active alert', 'capacity warning', 'critical alert', 'threshold']):
        eval_res = alert_system.evaluate_hospital_alerts(df_raw, execute_department_forecast)
        active_alerts = eval_res.get("active_alerts", [])
        thresholds = eval_res.get("thresholds", alert_system.load_thresholds())
        h_summary = eval_res.get("hospital_summary", {})

        alert_lines = ""
        if active_alerts:
            for alt in active_alerts[:6]:
                type_lbl = "Current Red-Zone Alert" if alt.get("trigger_type") == "CURRENT_OCCUPANCY" else "Predicted Red-Zone Warning"
                alert_lines += f"- **{alt['department']}** (`{type_lbl}`): {alt['occupancy_rate']}% occupancy | **{alt['available_beds']} beds available** out of {alt['total_beds']} (Time: {alt['timestamp']})\n"
        else:
            alert_lines = "- *No active red-zone alerts. All hospital departments are currently operating within safe capacity limits.*"

        resp = f"""### CarePulse Hospital Bed Occupancy Alert Status

**Configured Zone Thresholds:**
- **Green Zone:** < {thresholds['green_max']:.0f}% (Normal Operations)
- **Yellow Zone:** {thresholds['green_max']:.0f}% to < {thresholds['red_min']:.0f}% (Elevated Stress / Warning)
- **Red Zone:** ≥ {thresholds['red_min']:.0f}% (Critical Capacity Stress)

**Hospital-Wide Census:** **{h_summary.get('occupancy_rate', 0)}%** ({h_summary.get('occupied_beds', 0)}/{h_summary.get('total_beds', 0)} beds occupied | **{h_summary.get('available_beds', 0)} available beds**)

#### Active Critical Alerts ({len(active_alerts)} Total):
{alert_lines}

**Operational Recommendation:** Immediate capacity assessment and resource planning are recommended for any units in the Red Zone."""

        return {
            "answer": resp,
            "metrics": {
                "active_alerts_count": len(active_alerts),
                "thresholds": thresholds,
                "hospital_summary": h_summary
            },
            "suggestions": [
                "Forecast bed occupancy for Cardiology tomorrow",
                "How many beds are available in Surgery?",
                "What was the latest observed hospital census?"
            ]
        }

    # -------------------------------------------------------------
    # INTENT: Department Bed Occupancy / Availability / Peak Demand
    # -------------------------------------------------------------
    target_dept = dept if dept else "General Medicine"
    pred = execute_department_forecast(
        dept_name=target_dept,
        custom_adm=adm,
        custom_emerg=emerg,
        custom_disc=disc,
        season=query_info['season'],
        month=query_info['month'],
        is_holiday=query_info['is_holiday']
    )

    if pred:
        demand_badge = "[CRITICAL] PEAK DEMAND ALERT" if pred['is_peak_demand'] else "[OPTIMAL] ADEQUATE CAPACITY BUFFER"
        alert_recommendation = (
            "**Operational Action:** Immediate overflow protocol recommended. Consider deferring non-urgent elective admissions to prevent emergency boarding."
            if pred['is_peak_demand']
            else "**Operational Action:** Capacity buffer is adequate. Routine inpatient admissions and elective procedures can proceed as scheduled."
        )

        resp = f"""### Bed Occupancy Forecast: {target_dept}
**Status:** `{demand_badge}`

Based on current operational parameters and historical trends, here is the projected capacity allocation for **{target_dept}**:

* **Predicted Bed Occupancy:** **{pred['predicted_occupied_beds']} beds** (Projected Occupancy Rate: **{pred['predicted_occupancy_rate']}%**)
* **Estimated Available Beds:** **{pred['estimated_available_beds']} beds** (out of **{pred['total_bed_capacity']} total beds**)
* **Demand Classification:** **{pred['demand_level']}**
* **Inputs Evaluated:** {pred['inputs']['admissions']} Daily Admissions, {pred['inputs']['emergencies']} Emergency Cases, {pred['inputs']['discharges']} Discharges in {pred['inputs']['season']}.

{alert_recommendation}"""

        # Context-aware follow-up suggestions
        suggestions = []
        if not pred['is_peak_demand']:
            suggestions.append(f"What if emergency admissions in {target_dept} surge to 15?")
        else:
            suggestions.append(f"What if discharges in {target_dept} increase to 12?")

        alt_dept = "Surgery" if target_dept != "Surgery" else "Cardiology"
        suggestions.append(f"How many beds are available in {alt_dept}?")
        suggestions.append(f"Give me the 7-day outlook for {target_dept}")

        return {
            "answer": resp,
            "metrics": pred,
            "suggestions": suggestions
        }

    # -------------------------------------------------------------
    # FALLBACK: Out of Scope / General Guidance
    # -------------------------------------------------------------
    return {
        "answer": """### CarePulse Research Assistant

I can answer questions regarding **hospital bed occupancy forecasting, bed availability, emergency surges, and peak-demand risks** using verified operational hospital records and validated forecasting algorithms.

You can ask questions such as:
1. **Department Occupancy:** *"Forecast bed occupancy for Cardiology tomorrow"*
2. **Bed Availability:** *"How many beds are available in Surgery?"*
3. **What-If Clinical Surges:** *"Will General Medicine face peak demand with 15 admissions and 5 emergencies?"*
4. **Multi-Day Outlook:** *"What is the 7-day outlook for Pediatrics?"*
5. **Model Evaluation:** *"What is the system evaluation RMSE, MAE, and R² score?"*""",
        "metrics": None,
        "suggestions": [
            "Forecast bed occupancy for General Medicine",
            "How many beds are available in Cardiology?",
            "What is the model evaluation RMSE and R²?",
            "What is the latest observed hospital status?"
        ]
    }


# =============================================================================
# FLASK HTTP ROUTES
# =============================================================================

@app.route('/')
def home():
    """Serves the clean, conversational research-assistant interface."""
    return render_template('index.html')


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """
    Main conversational endpoint for user questions.
    Processes clinical questions through the CarePulse AI Agent, invoking
    trained ML models and MCP tools with deterministic clinical fallback.
    Accepts: { "question": str, "history": list }
    Returns: { "success": bool, "answer": str, "metrics": dict, "suggestions": list, "agent_mode": str, "tools_called": list }
    """
    data = request.get_json() or {}
    user_question = data.get('question', '').strip()
    history = data.get('history', [])

    if not user_question:
        return jsonify({
            "success": False,
            "error": "Question cannot be empty."
        }), 400

    flask_context = {
        "execute_department_forecast": execute_department_forecast,
        "df_raw": df_raw,
        "alert_system": alert_system,
        "dept_aliases": DEPT_ALIASES,
        "generate_research_answer": generate_research_answer,
        "parse_conversational_query": parse_conversational_query
    }

    try:
        agent_res = carepulse_agent.run_query(user_question, history, flask_context)
        if agent_res and agent_res.get("success"):
            return jsonify({
                "success": True,
                "question": user_question,
                "answer": agent_res.get("answer"),
                "metrics": agent_res.get("metrics"),
                "suggestions": agent_res.get("suggestions", []),
                "agent_mode": agent_res.get("agent_mode", "CarePulse AI Agent"),
                "tools_called": agent_res.get("tools_called", [])
            })
    except Exception as e:
        print(f"[CarePulse API] Agent run notice: {e}")

    # Fallback to direct conversational engine
    query_info = parse_conversational_query(user_question, history)
    response_data = generate_research_answer(query_info)

    return jsonify({
        "success": True,
        "question": user_question,
        "answer": response_data["answer"],
        "metrics": response_data["metrics"],
        "suggestions": response_data["suggestions"],
        "agent_mode": "CarePulse Local Engine",
        "tools_called": []
    })


# =============================================================================
# INTELLIGENT BED OCCUPANCY ALERT SYSTEM ENDPOINTS
# =============================================================================

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """
    Returns active alerts, previous alert history, current thresholds,
    and departmental zone statuses.
    """
    include_history = request.args.get('history', '1') == '1'
    eval_res = alert_system.evaluate_hospital_alerts(df_raw, execute_department_forecast)
    active = alert_system.alert_store.get_active_alerts()
    all_alerts = alert_system.alert_store.get_all_alerts() if include_history else active

    return jsonify({
        "success": True,
        "active_count": len(active),
        "active_alerts": active,
        "alerts_history": all_alerts,
        "thresholds": alert_system.load_thresholds(),
        "hospital_summary": eval_res.get("hospital_summary", {}),
        "departments": eval_res.get("departments", [])
    })


@app.route('/api/alerts/acknowledge', methods=['POST'])
def acknowledge_alert():
    """
    Marks an active alert as acknowledged by authorized hospital staff.
    Accepts: { "alert_id": str, "user": str }
    """
    data = request.get_json() or {}
    alert_id = data.get('alert_id', '').strip()
    user = data.get('user', 'Hospital Staff').strip()

    if not alert_id:
        return jsonify({"success": False, "error": "alert_id is required."}), 400

    ack_alert = alert_system.alert_store.acknowledge_alert(alert_id, user=user)
    if not ack_alert:
        return jsonify({"success": False, "error": f"Alert '{alert_id}' not found."}), 404

    return jsonify({
        "success": True,
        "message": f"Alert {alert_id} acknowledged by {user}.",
        "alert": ack_alert
    })


@app.route('/api/alerts/thresholds', methods=['GET', 'POST'])
def alert_thresholds():
    """
    GET: Returns current configured zone thresholds.
    POST: Updates thresholds with strict validation (0 < green_max < red_min <= 100).
    """
    if request.method == 'GET':
        return jsonify({
            "success": True,
            "thresholds": alert_system.load_thresholds()
        })

    data = request.get_json() or {}
    try:
        green_max = float(data.get('green_max'))
        red_min = float(data.get('red_min'))
        updated = alert_system.save_thresholds(green_max, red_min)
        # Re-evaluate with new thresholds
        alert_system.evaluate_hospital_alerts(df_raw, execute_department_forecast)
        return jsonify({
            "success": True,
            "message": "Zone thresholds updated successfully.",
            "thresholds": updated
        })
    except (ValueError, TypeError) as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.route('/api/occupancy-status', methods=['GET'])
def get_occupancy_status():
    """
    Evaluates real-time occupancy and classifies all departments into
    Green, Yellow, or Red zone.
    """
    status_data = alert_system.evaluate_hospital_alerts(df_raw, execute_department_forecast)
    return jsonify({
        "success": True,
        "status": status_data
    })


@app.route('/api/alerts/evaluate', methods=['POST'])
def trigger_alert_evaluation():
    """Manually triggers evaluation of hospital occupancy alerts."""
    status_data = alert_system.evaluate_hospital_alerts(df_raw, execute_department_forecast)
    return jsonify({
        "success": True,
        "message": "Alert evaluation completed.",
        "active_count": len(alert_system.alert_store.get_active_alerts()),
        "status": status_data
    })


# =============================================================================
# GRAPHICAL ANALYSIS ROUTES (PYTHON MATPLOTLIB)
# =============================================================================

@app.route('/api/charts-info', methods=['GET'])
def get_charts_info():
    """Returns metadata and descriptions for all 5 graphical analysis charts."""
    return jsonify({
        "success": True,
        "charts": charts.get_charts_catalog()
    })


@app.route('/api/chart/<chart_id>', methods=['GET'])
def get_chart_image(chart_id):
    """Serves high-resolution PNG visualizations generated via Matplotlib."""
    force_refresh = request.args.get('refresh') == '1' or request.args.get('force') == '1'
    img_bytes = charts.get_chart_bytes(chart_id, force=force_refresh)
    if img_bytes is None:
        return jsonify({"success": False, "error": f"Chart '{chart_id}' not found or could not be generated"}), 404

    return Response(
        img_bytes,
        mimetype='image/png',
        headers={
            'Cache-Control': 'no-cache, must-revalidate',
            'Content-Disposition': 'inline'
        }
    )


# Legacy & Automated Testing Endpoints (Preserved for full compatibility)
@app.route('/api/dashboard-overview', methods=['GET'])
def dashboard_overview():
    if df_raw.empty:
        return jsonify({"success": False, "error": "Dataset empty"}), 500

    latest_date = df_raw['Date'].max()
    latest_df = df_raw[df_raw['Date'] == latest_date]
    tot_beds = int(latest_df['Total_Hospital_Beds'].sum())
    tot_occ = int(latest_df['Beds_Occupied'].sum())

    return jsonify({
        "success": True,
        "latest_observation_date": str(latest_date.date()),
        "total_hospital_beds": tot_beds,
        "total_beds_occupied": tot_occ,
        "total_beds_available": max(0, tot_beds - tot_occ),
        "departments": latest_df.to_dict(orient='records'),
        "model_metadata": metadata
    })


@app.route('/api/forecast-horizon', methods=['GET'])
def forecast_horizon():
    if df_raw.empty:
        return jsonify({"success": False, "error": "Dataset empty"}), 500

    horizon_days = []
    latest_date = df_raw['Date'].max()
    for step in range(1, 8):
        target_date = latest_date + pd.Timedelta(days=step)
        depts_res = []
        for d in df_raw['Department'].unique():
            p = execute_department_forecast(d)
            if p:
                depts_res.append(p)
        horizon_days.append({
            "step": step,
            "forecast_date": str(target_date.date()),
            "departments": depts_res
        })

    return jsonify({
        "success": True,
        "horizon_days": 7,
        "daily_forecasts": horizon_days
    })


@app.route('/api/forecast-scenario', methods=['POST'])
def forecast_scenario():
    data = request.get_json() or {}
    dept = data.get('department', 'General Medicine')
    try:
        adm = float(data.get('expected_admissions', 8))
        emerg = float(data.get('expected_emergency', 3))
        disc = float(data.get('expected_discharges', 5))
    except (ValueError, TypeError) as e:
        return jsonify({"success": False, "error": str(e)}), 400

    res = execute_department_forecast(dept, custom_adm=adm, custom_emerg=emerg, custom_disc=disc)
    if not res:
        return jsonify({"success": False, "error": "Department not found"}), 404

    return jsonify({
        "success": True,
        "department": res["department"],
        "total_bed_capacity": res["total_bed_capacity"],
        "predicted_occupancy_rate": res["predicted_occupancy_rate"],
        "predicted_occupied_beds": res["predicted_occupied_beds"],
        "estimated_available_beds": res["estimated_available_beds"],
        "is_peak_demand": res["is_peak_demand"],
        "demand_level": res["demand_level"]
    })


# =============================================================================
# DEDICATED PREDICTION API FOR MCP SERVER & AI AGENT
# =============================================================================

FORECAST_HISTORY_FILE = "data/forecast_history.json"

def _append_forecast_history(entry):
    """Safely appends a forecast execution record to persistent audit storage."""
    try:
        history = []
        if os.path.exists(FORECAST_HISTORY_FILE):
            with open(FORECAST_HISTORY_FILE, 'r') as f:
                history = json.load(f)
        history.append(entry)
        if len(history) > 100:
            history = history[-100:]
        os.makedirs(os.path.dirname(FORECAST_HISTORY_FILE), exist_ok=True)
        with open(FORECAST_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save forecast history: {e}")


@app.route('/api/forecast', methods=['POST'])
def api_forecast():
    """
    Dedicated ML Prediction API Endpoint.
    Accepts validated inputs, constructs the exact 16-feature vector expected by
    the preprocessor and model, and returns verified model predictions.
    """
    if best_model is None or preprocessor is None:
        return jsonify({
            "success": False,
            "error": "Machine learning model or preprocessor is not loaded.",
            "model_status": "UNAVAILABLE"
        }), 500

    data = request.get_json() or {}
    raw_dept = data.get('department')
    if not raw_dept:
        return jsonify({
            "success": False,
            "error": "Missing required field 'department'.",
            "supported_departments": sorted(list(set(DEPT_ALIASES.values())))
        }), 400

    dept = DEPT_ALIASES.get(str(raw_dept).strip().lower(), str(raw_dept).strip())
    valid_depts = df_raw['Department'].unique().tolist() if not df_raw.empty else []
    if dept not in valid_depts:
        return jsonify({
            "success": False,
            "error": f"Department '{raw_dept}' not recognized.",
            "supported_departments": sorted(valid_depts)
        }), 404

    # Validate numeric parameters
    custom_adm = None
    custom_emerg = None
    custom_disc = None
    try:
        if 'expected_admissions' in data and data['expected_admissions'] is not None:
            custom_adm = float(data['expected_admissions'])
            if custom_adm < 0:
                raise ValueError("expected_admissions cannot be negative.")
        elif 'admissions' in data and data['admissions'] is not None:
            custom_adm = float(data['admissions'])
            if custom_adm < 0:
                raise ValueError("admissions cannot be negative.")

        if 'expected_emergency' in data and data['expected_emergency'] is not None:
            custom_emerg = float(data['expected_emergency'])
            if custom_emerg < 0:
                raise ValueError("expected_emergency cannot be negative.")
        elif 'emergencies' in data and data['emergencies'] is not None:
            custom_emerg = float(data['emergencies'])
            if custom_emerg < 0:
                raise ValueError("emergencies cannot be negative.")

        if 'expected_discharges' in data and data['expected_discharges'] is not None:
            custom_disc = float(data['expected_discharges'])
            if custom_disc < 0:
                raise ValueError("expected_discharges cannot be negative.")
        elif 'discharges' in data and data['discharges'] is not None:
            custom_disc = float(data['discharges'])
            if custom_disc < 0:
                raise ValueError("discharges cannot be negative.")

        season = data.get('season', 'Winter')
        if season not in ['Winter', 'Spring', 'Summer', 'Fall']:
            season = 'Winter'

        month = int(data.get('month', 1))
        if not (1 <= month <= 12):
            month = 1

        is_holiday = int(data.get('is_holiday', 0))
        is_holiday = 1 if is_holiday else 0
    except (ValueError, TypeError) as e:
        return jsonify({
            "success": False,
            "error": f"Invalid numerical input: {str(e)}"
        }), 400

    res = execute_department_forecast(
        dept_name=dept,
        custom_adm=custom_adm,
        custom_emerg=custom_emerg,
        custom_disc=custom_disc,
        season=season,
        month=month,
        is_holiday=is_holiday
    )

    if not res:
        return jsonify({
            "success": False,
            "error": f"Could not generate forecast for department '{dept}'. Check historical records."
        }), 500

    zone_info = alert_system.classify_occupancy_zone(res["predicted_occupancy_rate"])

    result_payload = {
        "success": True,
        "model_status": "ACTIVE_PRODUCTION_MODEL",
        "target_predicted": "Occupancy_Rate_Pct",
        "target_meaning": "Continuous hospital bed occupancy rate as a percentage (0-100%)",
        "department": res["department"],
        "predicted_occupancy_rate": res["predicted_occupancy_rate"],
        "total_hospital_beds": res["total_bed_capacity"],
        "predicted_occupied_beds": res["predicted_occupied_beds"],
        "estimated_available_beds": res["estimated_available_beds"],
        "is_peak_demand": res["is_peak_demand"],
        "demand_level": res["demand_level"],
        "zone": zone_info["zone"],
        "zone_label": zone_info["label"],
        "zone_severity": zone_info.get("severity", "normal"),
        "inputs_evaluated": res["inputs"],
        "model_metadata": {
            "algorithm": metadata.get("selected_model", "Linear Regression"),
            "validation_rmse": metadata.get("metrics", {}).get("Linear Regression", {}).get("rmse", 2.231),
            "validation_r2": metadata.get("metrics", {}).get("Linear Regression", {}).get("r2", 0.976),
            "training_dataset": "data/hospital_occupancy.csv (8,768 records)"
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "warnings": []
    }

    _append_forecast_history(result_payload)
    return jsonify(result_payload)


@app.route('/api/forecast/history', methods=['GET'])
def get_forecast_history_api():
    """Returns audit log of recent ML forecasts."""
    history = []
    if os.path.exists(FORECAST_HISTORY_FILE):
        try:
            with open(FORECAST_HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []
    return jsonify({
        "success": True,
        "count": len(history),
        "history": list(reversed(history[-50:]))
    })


@app.route('/api/department/<dept_name>/status', methods=['GET'])
def get_single_department_status(dept_name):
    """Returns verified capacity, latest census, and zone status for a specific department."""
    if df_raw.empty:
        return jsonify({"success": False, "error": "Dataset empty"}), 500

    dept = DEPT_ALIASES.get(str(dept_name).strip().lower(), str(dept_name).strip())
    dept_df = df_raw[df_raw['Department'] == dept]
    if dept_df.empty:
        return jsonify({
            "success": False,
            "error": f"Department '{dept_name}' not found.",
            "supported_departments": sorted(df_raw['Department'].unique().tolist())
        }), 404

    last_row = dept_df.sort_values('Date').iloc[-1]
    tot = int(last_row['Total_Hospital_Beds'])
    occ = int(last_row['Beds_Occupied'])
    rate = float(last_row['Occupancy_Rate_Pct'])
    avail = max(0, tot - occ)
    zone_info = alert_system.classify_occupancy_zone(rate)

    # ML Forecast
    pred = execute_department_forecast(dept)

    return jsonify({
        "success": True,
        "department": dept,
        "observation_date": str(last_row['Date'].date()),
        "total_hospital_beds": tot,
        "beds_occupied": occ,
        "available_beds": avail,
        "occupancy_rate_pct": rate,
        "zone_info": zone_info,
        "daily_admissions": int(last_row['Daily_Admissions']),
        "emergency_admissions": int(last_row['Emergency_Admissions']),
        "daily_discharges": int(last_row['Daily_Discharges']),
        "avg_length_of_stay": float(last_row['Avg_Length_of_Stay']),
        "forecast_next_day": pred
    })


@app.route('/api/historical-trends', methods=['GET'])
def historical_trends():
    if df_raw.empty:
        return jsonify({"success": False, "error": "Dataset empty"}), 500

    daily = df_raw.groupby('Date').agg(
        total_beds=('Total_Hospital_Beds', 'sum'),
        occupied=('Beds_Occupied', 'sum')
    ).reset_index().tail(60)

    daily['rate'] = round((daily['occupied'] / daily['total_beds']) * 100.0, 1)

    return jsonify({
        "success": True,
        "dates": [str(d.date()) for d in daily['Date']],
        "occupancy_rates": daily['rate'].tolist()
    })


@app.route('/api/model-evaluation', methods=['GET'])
def model_evaluation():
    return jsonify({
        "success": True,
        "selected_model": metadata.get('selected_model', 'Linear Regression'),
        "metrics": metadata.get('metrics', {})
    })


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)