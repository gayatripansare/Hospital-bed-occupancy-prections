/**
 * =============================================================================
 * CAREPULSE ANALYTICS - MODEL CONTEXT PROTOCOL (MCP) SERVER
 * Hospital Bed Occupancy Forecasting & Capacity Surveillance
 * =============================================================================
 * Exposes 6 standard MCP tools allowing conversational AI agents to interact
 * directly with the Python Flask backend and trained machine learning model.
 * 
 * Strict Constraint: Numerical predictions are obtained solely from the trained
 * ML model (best_model.pkl). The agent and MCP tools never fabricate or guess numbers.
 * =============================================================================
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import { backendClient } from './client.js';

// Initialize MCP Server
const server = new McpServer({
  name: 'carepulse-hospital-bed-forecasting',
  version: '1.0.0',
});

// =============================================================================
// TOOL 1: predict_hospital_demand
// =============================================================================
server.tool(
  'predict_hospital_demand',
  'Calls the Flask prediction API and returns the actual bed occupancy prediction from the trained Scikit-Learn ML model. Returns predicted occupancy percentage, estimated available beds, and surge indicators.',
  {
    department: z
      .string()
      .describe('Clinical department name (e.g., Cardiology, General Medicine, Neurology, Pediatrics, Orthopedics, Oncology, Surgery, Obstetrics & Gynecology)'),
    expected_admissions: z
      .number()
      .min(0)
      .optional()
      .describe('Expected daily admissions count (optional, uses historical baseline if omitted)'),
    expected_emergency: z
      .number()
      .min(0)
      .optional()
      .describe('Expected emergency arrivals count (optional, uses historical baseline if omitted)'),
    expected_discharges: z
      .number()
      .min(0)
      .optional()
      .describe('Expected daily patient discharges (optional, uses historical baseline if omitted)'),
    season: z
      .enum(['Winter', 'Spring', 'Summer', 'Fall'])
      .optional()
      .describe('Season of the year (default: Winter)'),
    month: z
      .number()
      .min(1)
      .max(12)
      .optional()
      .describe('Month index (1-12)'),
    is_holiday: z
      .number()
      .min(0)
      .max(1)
      .optional()
      .describe('1 if date is a public holiday, 0 otherwise'),
  },
  async (args) => {
    try {
      const result = await backendClient.predictHospitalDemand(args);
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
      };
    } catch (err: any) {
      return {
        content: [{ type: 'text', text: JSON.stringify({ success: false, error: err.message }) }],
        isError: true,
      };
    }
  }
);

// =============================================================================
// TOOL 2: get_current_occupancy
// =============================================================================
server.tool(
  'get_current_occupancy',
  'Retrieves verified current hospital census data from real hospital observations. Returns total hospital bed capacity, occupied beds, available beds, overall occupancy rate, and zone classifications. Does NOT derive occupancy from admissions alone.',
  {},
  async () => {
    try {
      const result = await backendClient.getCurrentOccupancy();
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
      };
    } catch (err: any) {
      return {
        content: [{ type: 'text', text: JSON.stringify({ success: false, error: err.message }) }],
        isError: true,
      };
    }
  }
);

// =============================================================================
// TOOL 3: get_department_status
// =============================================================================
server.tool(
  'get_department_status',
  'Retrieves verified capacity, latest census observations, available beds, and next-day ML forecast for a specific clinical department.',
  {
    department: z
      .string()
      .describe('Name of clinical department (e.g., Cardiology, Surgery, General Medicine, Pediatrics, etc.)'),
  },
  async ({ department }) => {
    try {
      const result = await backendClient.getDepartmentStatus(department);
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
      };
    } catch (err: any) {
      return {
        content: [{ type: 'text', text: JSON.stringify({ success: false, error: err.message }) }],
        isError: true,
      };
    }
  }
);

// =============================================================================
// TOOL 4: check_red_zone
// =============================================================================
server.tool(
  'check_red_zone',
  'Evaluates a validated occupancy percentage against configurable hospital capacity thresholds (Green: <70%, Yellow: 70-85%, Red: >=85%). Returns severity, alert status, and clinical operational recommendations.',
  {
    occupancy_rate: z
      .number()
      .min(0)
      .max(100)
      .describe('The continuous bed occupancy percentage (0.0 to 100.0) to evaluate'),
    department: z
      .string()
      .optional()
      .describe('Optional department name being evaluated'),
  },
  async ({ occupancy_rate, department }) => {
    try {
      const result = await backendClient.checkRedZone(occupancy_rate, department);
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
      };
    } catch (err: any) {
      return {
        content: [{ type: 'text', text: JSON.stringify({ success: false, error: err.message }) }],
        isError: true,
      };
    }
  }
);

// =============================================================================
// TOOL 5: get_active_alerts
// =============================================================================
server.tool(
  'get_active_alerts',
  'Retrieves all active Red-Zone capacity alerts and early-warning triggers from the application alert storage. Includes department name, current vs. predicted trigger reason, occupancy rate, and available beds.',
  {},
  async () => {
    try {
      const result = await backendClient.getActiveAlerts();
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
      };
    } catch (err: any) {
      return {
        content: [{ type: 'text', text: JSON.stringify({ success: false, error: err.message }) }],
        isError: true,
      };
    }
  }
);

// =============================================================================
// TOOL 6: get_forecast_history
// =============================================================================
server.tool(
  'get_forecast_history',
  'Retrieves the recent audit trail of ML forecasts executed by the application backend.',
  {
    limit: z
      .number()
      .min(1)
      .max(50)
      .optional()
      .describe('Maximum number of historical forecast records to retrieve (default: 10)'),
  },
  async ({ limit }) => {
    try {
      const result = await backendClient.getForecastHistory(limit || 10);
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
      };
    } catch (err: any) {
      return {
        content: [{ type: 'text', text: JSON.stringify({ success: false, error: err.message }) }],
        isError: true,
      };
    }
  }
);

// Start MCP Server over stdio
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('[CarePulse MCP Server] Connected and running over stdio transport.');
}

main().catch((err) => {
  console.error('[CarePulse MCP Server] Fatal startup error:', err);
  process.exit(1);
});
