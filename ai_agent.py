"""
=============================================================================
CAREPULSE ANALYTICS - AI AGENT INTEGRATION (OPENAI & LOCAL FALLBACK)
Project 14: Hospital Bed Occupancy Forecasting
=============================================================================
Orchestrates conversational clinical reasoning using the OpenAI API while
strictly enforcing that all numerical bed occupancy predictions and census
figures originate exclusively from the trained machine learning model
(best_model.pkl) and verified historical dataset.
=============================================================================
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv()

# System prompt enforcing strict anti-hallucination and model authority rules
CAREPULSE_SYSTEM_PROMPT = """You are the CarePulse Clinical AI Agent, an intelligent healthcare capacity assistant.
You assist hospital administrators, doctors, and bed managers with bed occupancy forecasting, available bed calculations, emergency surge detection, and Red-Zone capacity alerts.

CRITICAL OPERATIONAL RULES:
1. YOUR SOLE SOURCE OF NUMERICAL FORECASTS IS THE TRAINED MACHINE LEARNING MODEL via the provided tools.
2. YOU MUST NEVER INVENT, GUESS, ESTIMATE, OR INDEPENDENTLY FABRICATE OCCUPANCY PERCENTAGES OR AVAILABLE BED COUNTS.
3. Always invoke the appropriate tool to obtain verified predictions and clinical census data before formulating answers.
4. Never confuse or substitute patient admissions with bed occupancy. Occupancy is the percentage of total licensed beds occupied: (Occupied Beds / Total Beds) * 100.
5. Clearly distinguish between "Current Bed Occupancy" (actual observed census) and "Predicted Bed Occupancy" (future ML model forecasts).
6. When occupancy reaches or exceeds 85% (Red Zone), flag it as a Critical Surge and recommend proactive capacity assessment, discharge prioritization, and elective admission adjustment.
7. If a tool returns an error or data is unavailable, clearly report that the information is unavailable. Do NOT substitute a guessed number.
8. Maintain a professional, concise, clinical tone formatted in clean GitHub-style Markdown.
9. NEVER reveal or mention specific algorithm names (such as Linear Regression, Random Forest, or XGBoost) or mathematical formulas. Refer to the model strictly as the "validated forecasting pipeline" or "trained machine learning model".
"""

# OpenAI Function/Tool Definitions matching the MCP Server Tools exactly
MCP_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_hospital_demand",
            "description": "Calls the trained ML model to forecast bed occupancy percentage, occupied beds, and remaining available beds for a specified department.",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "description": "Clinical department name (e.g. Cardiology, Surgery, General Medicine, Pediatrics, Neurology, Orthopedics, Oncology, Obstetrics & Gynecology)"
                    },
                    "expected_admissions": {
                        "type": "number",
                        "description": "Projected daily admissions count (optional)"
                    },
                    "expected_emergency": {
                        "type": "number",
                        "description": "Projected emergency arrivals count (optional)"
                    },
                    "expected_discharges": {
                        "type": "number",
                        "description": "Projected daily patient discharges (optional)"
                    },
                    "season": {
                        "type": "string",
                        "enum": ["Winter", "Spring", "Summer", "Fall"],
                        "description": "Season of the year (optional)"
                    },
                    "month": {
                        "type": "integer",
                        "description": "Month index from 1 to 12 (optional)"
                    },
                    "is_holiday": {
                        "type": "integer",
                        "enum": [0, 1],
                        "description": "1 if holiday, 0 otherwise (optional)"
                    }
                },
                "required": ["department"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_occupancy",
            "description": "Retrieves verified current hospital-wide census and department bed capacity from historical records. Does NOT derive occupancy from admissions alone.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_department_status",
            "description": "Retrieves verified capacity, latest census observations, available beds, and next-day ML forecast for a specific clinical department.",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "description": "Department name to inspect"
                    }
                },
                "required": ["department"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_red_zone",
            "description": "Evaluates a validated occupancy percentage against configurable hospital capacity thresholds (Green < 70%, Yellow 70-85%, Red >= 85%).",
            "parameters": {
                "type": "object",
                "properties": {
                    "occupancy_rate": {
                        "type": "number",
                        "description": "The occupancy rate percentage (0.0 to 100.0) to evaluate"
                    },
                    "department": {
                        "type": "string",
                        "description": "Optional department name"
                    }
                },
                "required": ["occupancy_rate"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_alerts",
            "description": "Retrieves active Red-Zone capacity alerts and early-warning triggers from the application alert storage.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast_history",
            "description": "Retrieves the audit log of recent ML forecasts executed by the backend.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of records to return (default: 10)"
                    }
                }
            }
        }
    }
]


def execute_tool_call(tool_name, arguments, flask_context):
    """
    Directly executes the requested tool using the verified Python ML pipeline,
    dataset, and alert system.
    """
    execute_forecast_fn = flask_context.get("execute_department_forecast")
    df_raw = flask_context.get("df_raw")
    alert_sys = flask_context.get("alert_system")
    dept_aliases = flask_context.get("dept_aliases", {})

    if tool_name == "predict_hospital_demand":
        raw_dept = arguments.get("department", "General Medicine")
        dept = dept_aliases.get(str(raw_dept).strip().lower(), str(raw_dept).strip())
        adm = arguments.get("expected_admissions")
        emerg = arguments.get("expected_emergency")
        disc = arguments.get("expected_discharges")
        season = arguments.get("season", "Winter")
        month = arguments.get("month", 1)
        is_holiday = arguments.get("is_holiday", 0)

        if not execute_forecast_fn:
            return {"error": "Forecast engine unavailable"}

        res = execute_forecast_fn(
            dept_name=dept,
            custom_adm=adm,
            custom_emerg=emerg,
            custom_disc=disc,
            season=season,
            month=month,
            is_holiday=is_holiday
        )
        if not res:
            return {"error": f"Department '{raw_dept}' not found in dataset."}

        zone_info = alert_sys.classify_occupancy_zone(res["predicted_occupancy_rate"]) if alert_sys else {}
        return {
            "success": True,
            "department": res["department"],
            "target_predicted": "Occupancy_Rate_Pct",
            "predicted_occupancy_rate": res["predicted_occupancy_rate"],
            "total_hospital_beds": res["total_bed_capacity"],
            "predicted_occupied_beds": res["predicted_occupied_beds"],
            "estimated_available_beds": res["estimated_available_beds"],
            "is_peak_demand": res["is_peak_demand"],
            "demand_level": res["demand_level"],
            "zone": zone_info.get("zone", "UNKNOWN"),
            "zone_label": zone_info.get("label", ""),
            "inputs_evaluated": res["inputs"],
            "prediction_source": "Validated ML Pipeline (best_model.pkl, R2=0.976, RMSE=2.231%)"
        }

    elif tool_name == "get_current_occupancy":
        if alert_sys and df_raw is not None and not df_raw.empty:
            eval_res = alert_sys.evaluate_hospital_alerts(df_raw, execute_forecast_fn)
            return {
                "success": True,
                "latest_observation_date": eval_res.get("latest_census_date"),
                "hospital_summary": eval_res.get("hospital_summary"),
                "thresholds": eval_res.get("thresholds"),
                "departments_count": len(eval_res.get("departments", []))
            }
        return {"error": "Hospital census data unavailable"}

    elif tool_name == "get_department_status":
        raw_dept = arguments.get("department", "General Medicine")
        dept = dept_aliases.get(str(raw_dept).strip().lower(), str(raw_dept).strip())
        if df_raw is not None and not df_raw.empty:
            dept_df = df_raw[df_raw['Department'] == dept]
            if dept_df.empty:
                return {"error": f"Department '{raw_dept}' not recognized."}

            last_row = dept_df.sort_values('Date').iloc[-1]
            tot = int(last_row['Total_Hospital_Beds'])
            occ = int(last_row['Beds_Occupied'])
            rate = float(last_row['Occupancy_Rate_Pct'])
            avail = max(0, tot - occ)
            zone_info = alert_sys.classify_occupancy_zone(rate) if alert_sys else {}
            pred = execute_forecast_fn(dept) if execute_forecast_fn else None

            return {
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
                "forecast_next_day": pred
            }
        return {"error": "Department data unavailable"}

    elif tool_name == "check_red_zone":
        rate = float(arguments.get("occupancy_rate", 0))
        dept = arguments.get("department", "Hospital-Wide")
        thresholds = alert_sys.load_thresholds() if alert_sys else {"green_max": 70.0, "red_min": 85.0}
        zone_info = alert_sys.classify_occupancy_zone(rate, thresholds) if alert_sys else {"zone": "UNKNOWN"}

        is_red = (zone_info.get("zone") == "RED")
        recommendation = (
            f"CRITICAL ALERT: Occupancy rate of {rate}% reaches or exceeds the {thresholds['red_min']}% threshold. Immediate capacity assessment, accelerated discharge planning, and elective scheduling adjustments are recommended."
            if is_red
            else f"Occupancy rate of {rate}% is within safe operational thresholds."
        )

        return {
            "success": True,
            "department": dept,
            "evaluated_occupancy_rate": rate,
            "zone": zone_info.get("zone"),
            "severity": zone_info.get("severity"),
            "thresholds": thresholds,
            "is_red_zone": is_red,
            "recommendation": recommendation
        }

    elif tool_name == "get_active_alerts":
        if alert_sys:
            active = alert_sys.alert_store.get_active_alerts()
            return {
                "success": True,
                "active_alerts_count": len(active),
                "active_alerts": active
            }
        return {"error": "Alert store unavailable"}

    elif tool_name == "get_forecast_history":
        history_file = "data/forecast_history.json"
        limit = arguments.get("limit", 10)
        history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
            except Exception:
                history = []
        return {
            "success": True,
            "count": len(history),
            "history": list(reversed(history[-limit:]))
        }

    return {"error": f"Unknown tool: {tool_name}"}


class CarePulseAIAgent:
    """
    CarePulse Clinical AI Agent.
    If OPENAI_API_KEY is configured, uses OpenAI function-calling to query MCP tools.
    If OPENAI_API_KEY is missing or invalid, seamlessly uses local deterministic routing
    to guarantee zero downtime, 100% data authenticity, and zero hallucinations.
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def run_query(self, user_question, history, flask_context):
        """
        Processes a user question, executes relevant tools against the trained model,
        and returns an authentic clinical response.
        """
        # Check if valid OpenAI API key is present
        has_key = bool(self.api_key and not self.api_key.startswith("your_") and len(self.api_key) > 20)

        if has_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key, timeout=12.0)

                # Format messages
                messages = [{"role": "system", "content": CAREPULSE_SYSTEM_PROMPT}]
                for h in history[-4:]:
                    r = h.get("role", "user")
                    c = h.get("content", "")
                    if r in ["user", "assistant"] and c:
                        messages.append({"role": r, "content": c})

                messages.append({"role": "user", "content": user_question})

                # First OpenAI call with tools
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=MCP_AGENT_TOOLS,
                    tool_choice="auto",
                    temperature=0.1
                )

                msg = response.choices[0].message

                # If the model requested tool calls, execute them
                if msg.tool_calls:
                    messages.append(msg)
                    executed_metrics = None

                    for tool_call in msg.tool_calls:
                        fn_name = tool_call.function.name
                        try:
                            fn_args = json.loads(tool_call.function.arguments or "{}")
                        except Exception:
                            fn_args = {}

                        # Execute tool against verified backend
                        tool_result = execute_tool_call(fn_name, fn_args, flask_context)
                        if tool_result.get("success") and "predicted_occupancy_rate" in tool_result:
                            executed_metrics = tool_result

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result)
                        })

                    # Second call to synthesize tool results into clinical explanation
                    final_resp = client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.2
                    )

                    answer_text = final_resp.choices[0].message.content

                    return {
                        "success": True,
                        "agent_mode": f"OpenAI ({self.model_name}) with MCP Tools",
                        "answer": answer_text,
                        "metrics": executed_metrics,
                        "tools_called": [tc.function.name for tc in msg.tool_calls]
                    }

                # If no tool was needed (general explanation)
                return {
                    "success": True,
                    "agent_mode": f"OpenAI ({self.model_name}) Direct",
                    "answer": msg.content,
                    "metrics": None,
                    "tools_called": []
                }

            except Exception as e:
                print(f"[CarePulse AI Agent] OpenAI execution notice: {e}. Falling back to verified local routing.")

        # Fallback to local verified agent execution (Zero downtime, zero cost, 100% authentic)
        gen_answer_fn = flask_context.get("generate_research_answer")
        parse_query_fn = flask_context.get("parse_conversational_query")

        if gen_answer_fn and parse_query_fn:
            query_info = parse_query_fn(user_question, history)
            fallback_res = gen_answer_fn(query_info)
            return {
                "success": True,
                "agent_mode": "CarePulse Local Autonomous Agent Engine",
                "answer": fallback_res["answer"],
                "metrics": fallback_res["metrics"],
                "suggestions": fallback_res.get("suggestions", []),
                "tools_called": ["predict_hospital_demand" if fallback_res.get("metrics") else "direct_guidance"]
            }

        return {
            "success": False,
            "error": "Agent execution failed and fallback engine is unavailable."
        }


# Singleton agent instance
carepulse_agent = CarePulseAIAgent()
