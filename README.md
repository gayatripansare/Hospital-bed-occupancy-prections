# CarePulse Analytics Platform
## Hospital Bed Occupancy Forecasting System with MCP Integration & AI Agent

---

## 1. Project Overview & Architecture

**CarePulse Analytics** is an enterprise-grade clinical capacity management and bed occupancy forecasting platform. It pairs a validated machine learning regression pipeline with a **Model Context Protocol (MCP)** server and an **AI Clinical Agent** (`gpt-4o-mini` / local fallback) to provide reliable, real-time inpatient capacity intelligence.

### Core Architectural Principle: The ML Model is the Single Source of Truth
The AI Agent and MCP Server **never invent, estimate, or hallucinate** numerical occupancy rates, patient census, or bed counts. Every numerical forecast is computed exclusively by the trained Scikit-Learn regression model (`model/best_model.pkl`) using autoregressive hospital records (`data/hospital_occupancy.csv`).

```
                    ┌─────────────────────────┐
                    │   CarePulse Web UI      │
                    │ (Charcoal & Violet UI)  │
                    └───────────┬─────────────┘
                                │ HTTP / REST
                                ▼
                    ┌─────────────────────────┐
                    │      Flask Backend      │
                    │       (app.py)          │
                    └──────┬───────────┬──────┘
                           │           │
            ┌──────────────┴──────┐    └──────────────┐
            ▼                     ▼                   ▼
┌───────────────────────┐ ┌───────────────┐ ┌───────────────────┐
│ CarePulse AI Agent    │ │ Alert System  │ │ Matplotlib Visuals│
│    (ai_agent.py)      │ │(alert_system) │ │   (charts.py)     │
└───────────┬───────────┘ └───────┬───────┘ └───────────────────┘
            │                     │
            ▼ JSON-RPC (stdio)    │
┌───────────────────────┐         │
│ TypeScript MCP Server │         │
│     (mcp-server/)     │         │
└───────────┬───────────┘         │
            │                     │
            └──────────┬──────────┘
                       ▼
         ┌───────────────────────────┐
         │  Trained ML Model & Data  │
         │ (best_model.pkl, preproc) │
         │ (data/hospital_occupancy) │
         └───────────────────────────┘
```

---

## 2. Key Capabilities & Features

1. **Conversational Research Assistant:**
   - Real-time natural language inquiries for bed occupancy forecasts, remaining available beds, and emergency surge stress testing.
   - Dual-mode agent reasoning: uses **OpenAI Function Calling** (`gpt-4o-mini`) via MCP tool schemas when an API key is provided, or seamlessly switches to a local deterministic clinical router when offline or running zero-cost.
   - Contextual conversation memory across multi-turn interactions.
   - Strict model name secrecy: technical algorithm names are abstracted to preserve clinical focus.

2. **Model Context Protocol (MCP) Server:**
   - Built on the official `@modelcontextprotocol/sdk` (Node.js & TypeScript).
   - Communicates with client agents via standard JSON-RPC 2.0 over `stdio`.
   - Exposes 6 clinical tools connecting to the Python ML backend:
     - `predict_hospital_demand`: Calls `best_model.pkl` for bed occupancy %, occupied beds, and remaining beds.
     - `get_current_occupancy`: Retrieves observed hospital-wide census and departmental status.
     - `get_department_status`: Department capacity, latest census observations, and next-day forecast.
     - `check_red_zone`: Evaluates occupancy against configurable capacity zone thresholds.
     - `get_active_alerts`: Retrieves active Red-Zone capacity alerts.
     - `get_forecast_history`: Returns persisted audit log of recent forecasts.

3. **Intelligent Bed Occupancy Alert System:**
   - Configurable capacity zones:
     - **Green Zone:** $< 70\%$ occupancy (Normal Operations)
     - **Yellow Zone:** $70\% - < 85\%$ occupancy (Elevated Demand)
     - **Red Zone:** $\ge 85\%$ occupancy (Critical Surge Risk)
   - Real-time automated alerts distinguishing **Current Red-Zone Alerts** from **Forecast-Based Early Warnings**.
   - Interactive Alert Center in UI for acknowledging alerts and adjusting thresholds.

4. **Graphical Analysis:**
   - 5 Matplotlib visualizations rendered via server-side cached PNG streams:
     - Historical Bed Occupancy (Purple & Lavender)
     - Department-wise Bed Occupancy (Green, Teal & Violet)
     - Admissions vs. Discharges (Orange & Emerald)
     - Emergency Cases Trend (Red & Coral)
     - Actual vs. Predicted Validation (Orange & Purple Scatter)

---

## 3. Machine Learning Model Performance

- **Target Variable:** Continuous `Occupancy_Rate_Pct` (0.0% to 100.0%)
- **Dataset:** `data/hospital_occupancy.csv` (8,768 daily observations across 8 departments, 2022–2024)
- **Mathematical Formulations:**
  $$\text{Predicted Occupied Beds} = \text{round}\left(\frac{\text{Occupancy Rate Pct} \times \text{Department Beds}}{100}\right)$$
  $$\text{Estimated Available Beds} = \max(0, \text{Department Beds} - \text{Predicted Occupied Beds})$$

### Official Held-Out Test Evaluation (Chronological Split):
| Model Pipeline | RMSE (%) | MAE (%) | $R^2$ Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| Naive Persistence Baseline | 7.366 | 5.206 | 0.734 | Benchmark |
| **Linear Regression Regressor** | **2.231** | **1.828** | **0.976** | **Selected Winner** |
| Random Forest Regressor | 2.947 | 1.938 | 0.957 | Candidate |
| XGBoost Regressor | 2.762 | 2.062 | 0.963 | Candidate |

---

## 4. Environment Setup & Configuration

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### Installation

1. **Python Dependencies:**
```powershell
pip install -r requirements.txt
```

2. **MCP Server Dependencies & Build:**
```powershell
cd mcp-server
npm install
npm run build
cd ..
```

3. **Environment Configuration:**
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```
Edit `.env` to configure your OpenAI key (optional):
```ini
FLASK_API_URL=http://127.0.0.1:5000
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```
*(If `OPENAI_API_KEY` is omitted, the application runs entirely in local autonomous agent mode with zero cost and zero external calls).*

---

## 5. Running the Application

### 1. Launch the Flask Backend
```powershell
python app.py
```
Open your browser at `http://127.0.0.1:5000/`.

### 2. Run the MCP Server Standalone / In Claude Desktop / In Agent Clients
```powershell
node mcp-server/dist/index.js
```
To configure in Claude Desktop (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "carepulse": {
      "command": "node",
      "args": ["C:/Users/LENOVO/OneDrive/Desktop/AIML Project/mcp-server/dist/index.js"],
      "env": {
        "FLASK_API_URL": "http://127.0.0.1:5000"
      }
    }
  }
}
```

---

## 6. Automated Verification Test Suites

CarePulse includes 3 test suites covering all 33 regression and integration test cases:

1. **MCP Server & AI Agent Integration Suite:**
```powershell
python test_mcp_integration.py
```
*(8/8 tests pass: API endpoints, stdio JSON-RPC handshake, tool execution, model authority, anti-hallucination)*

2. **Core Flask API & Charts Suite:**
```powershell
python test_api.py
```
*(14/14 tests pass: UI loading, forecasting, availability, memory, anonymized metrics, 5 charts)*

3. **Alert System Suite:**
```powershell
python test_alerts.py
```
*(11/11 tests pass: Green/Yellow/Red zone boundaries, early warnings, threshold configuration, deduplication)*
