/**
 * CarePulse Analytics - Flask Backend REST Client for MCP Server
 * Connects MCP tools directly to the Python Flask backend and trained ML model.
 */

import dotenv from 'dotenv';
dotenv.config();

const FLASK_URL = process.env.FLASK_API_URL || 'http://127.0.0.1:5000';
const TIMEOUT_MS = parseInt(process.env.FLASK_TIMEOUT_MS || '10000', 10);

async function requestWithTimeout(url: string, options: RequestInit = {}): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return res;
  } finally {
    clearTimeout(timeout);
  }
}

export class FlaskBackendClient {
  private baseUrl: string;

  constructor(baseUrl: string = FLASK_URL) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  /**
   * Tool 1: predict_hospital_demand
   * Calls POST /api/forecast to obtain predictions from best_model.pkl
   */
  async predictHospitalDemand(params: {
    department: string;
    expected_admissions?: number;
    expected_emergency?: number;
    expected_discharges?: number;
    season?: string;
    month?: number;
    is_holiday?: number;
  }) {
    const res = await requestWithTimeout(`${this.baseUrl}/api/forecast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || `Forecast API returned status ${res.status}`);
    }
    return data;
  }

  /**
   * Tool 2: get_current_occupancy
   * Retrieves verified census from GET /api/occupancy-status
   */
  async getCurrentOccupancy() {
    const res = await requestWithTimeout(`${this.baseUrl}/api/occupancy-status`, {
      method: 'GET',
    });

    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.error || `Occupancy status API returned status ${res.status}`);
    }
    return data.status;
  }

  /**
   * Tool 3: get_department_status
   * Retrieves specific department status from GET /api/department/<name>/status
   */
  async getDepartmentStatus(department: string) {
    const encoded = encodeURIComponent(department);
    const res = await requestWithTimeout(`${this.baseUrl}/api/department/${encoded}/status`, {
      method: 'GET',
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || `Department status API returned status ${res.status}`);
    }
    return data;
  }

  /**
   * Tool 4: check_red_zone
   * Queries configured thresholds from GET /api/alerts/thresholds and evaluates zone
   */
  async checkRedZone(occupancyRate: number, department?: string) {
    let greenMax = 70.0;
    let redMin = 85.0;

    try {
      const res = await requestWithTimeout(`${this.baseUrl}/api/alerts/thresholds`, {
        method: 'GET',
      });
      if (res.ok) {
        const data = await res.json();
        if (data.thresholds) {
          greenMax = data.thresholds.green_max ?? greenMax;
          redMin = data.thresholds.red_min ?? redMin;
        }
      }
    } catch (e) {
      // Gracefully fall back to standard project defaults: 70% and 85%
    }

    let zone = 'GREEN';
    let severity = 'normal';
    let recommendation = 'Capacity buffer is adequate. Routine operations can proceed.';

    if (occupancyRate >= redMin) {
      zone = 'RED';
      severity = 'critical';
      recommendation = `CRITICAL ALERT: Occupancy rate of ${occupancyRate}% reaches or exceeds the ${redMin}% threshold. Immediate capacity assessment, accelerated discharge planning, and elective scheduling adjustments are recommended.`;
    } else if (occupancyRate >= greenMax) {
      zone = 'YELLOW';
      severity = 'warning';
      recommendation = `WARNING: Occupancy rate of ${occupancyRate}% is elevated within the ${greenMax}%-${redMin}% range. Close monitoring and discharge preparation are advised.`;
    }

    return {
      success: true,
      department: department || 'Hospital-Wide',
      evaluated_occupancy_rate: occupancyRate,
      zone,
      severity,
      thresholds: {
        green_zone: `< ${greenMax}%`,
        yellow_zone: `${greenMax}% to < ${redMin}%`,
        red_zone: `≥ ${redMin}%`,
      },
      is_red_zone: zone === 'RED',
      recommendation,
    };
  }

  /**
   * Tool 5: get_active_alerts
   * Retrieves active alerts from GET /api/alerts
   */
  async getActiveAlerts() {
    const res = await requestWithTimeout(`${this.baseUrl}/api/alerts`, {
      method: 'GET',
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || `Alerts API returned status ${res.status}`);
    }

    return {
      success: true,
      active_count: data.active_count,
      active_alerts: data.active_alerts,
      thresholds: data.thresholds,
      evaluation_time: data.hospital_summary ? new Date().toISOString() : null,
    };
  }

  /**
   * Tool 6: get_forecast_history
   * Retrieves stored forecast records from GET /api/forecast/history
   */
  async getForecastHistory(limit: number = 10) {
    const res = await requestWithTimeout(`${this.baseUrl}/api/forecast/history`, {
      method: 'GET',
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || `Forecast history API returned status ${res.status}`);
    }

    const history = (data.history || []).slice(0, limit);
    return {
      success: true,
      total_recorded: data.count,
      retrieved_count: history.length,
      history,
    };
  }
}

export const backendClient = new FlaskBackendClient();
