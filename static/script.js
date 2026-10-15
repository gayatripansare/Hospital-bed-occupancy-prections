/**
 * =============================================================================
 * CAREPULSE ANALYTICS - RESEARCH ASSISTANT & GRAPHICAL ANALYSIS
 * Client Controller & Conversational Interface
 * =============================================================================
 */

let conversationHistory = [];

document.addEventListener('DOMContentLoaded', () => {
  const inputEl = document.getElementById('questionInput');

  // Auto-resize textarea as user types
  if (inputEl) {
    inputEl.addEventListener('input', function () {
      this.style.height = 'auto';
      this.style.height = `${Math.min(this.scrollHeight, 130)}px`;
    });

    // Enter submits (Shift+Enter inserts newline)
    inputEl.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        document.getElementById('questionForm').requestSubmit();
      }
    });
  }

  // Example Prompt Chips
  document.querySelectorAll('.chip-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const query = btn.getAttribute('data-query');
      if (query) {
        submitUserQuestion(query);
      }
    });
  });

  // Evaluate alerts & display banner if critical
  checkAlertsBanner();
  setInterval(checkAlertsBanner, 30000);
});

// =============================================================================
// TAB SWITCHING (RESEARCH ASSISTANT <-> GRAPHICAL ANALYSIS)
// =============================================================================

const CHART_IDS = [
  'historical_occupancy',
  'department_occupancy',
  'admissions_discharges',
  'emergency_cases',
  'actual_vs_predicted'
];

function switchTab(tabId) {
  const navAssistant = document.getElementById('navAssistant');
  const navGraphics = document.getElementById('navGraphics');
  const navAlerts = document.getElementById('navAlerts');
  const screenAssistant = document.getElementById('screenAssistant');
  const screenGraphics = document.getElementById('screenGraphics');
  const screenAlerts = document.getElementById('screenAlerts');
  const pageTitle = document.getElementById('pageTitle');

  // Clear active states
  [navAssistant, navGraphics, navAlerts].forEach(n => { if (n) n.classList.remove('active'); });
  [screenAssistant, screenGraphics, screenAlerts].forEach(s => { if (s) s.classList.remove('active'); });

  if (tabId === 'assistant') {
    if (navAssistant) navAssistant.classList.add('active');
    if (screenAssistant) screenAssistant.classList.add('active');
    if (pageTitle) pageTitle.textContent = 'CarePulse Research Assistant';
    scrollToBottom();
  } else if (tabId === 'graphics') {
    if (navGraphics) navGraphics.classList.add('active');
    if (screenGraphics) screenGraphics.classList.add('active');
    if (pageTitle) pageTitle.textContent = 'Hospital Inpatient Graphical Analysis';

    // Verify each chart image state upon entering Graphical Analysis
    CHART_IDS.forEach(chartId => {
      const imgEl = document.getElementById(`chartImg_${chartId}`);
      if (imgEl) {
        if (imgEl.complete && imgEl.naturalWidth > 0) {
          handleChartLoaded(chartId);
        } else if (imgEl.complete && imgEl.naturalWidth === 0) {
          retryChart(chartId);
        }
      }
    });
  } else if (tabId === 'alerts') {
    if (navAlerts) navAlerts.classList.add('active');
    if (screenAlerts) screenAlerts.classList.add('active');
    if (pageTitle) pageTitle.textContent = 'Bed Occupancy Alert Management';
    fetchAlerts();
  }
}

// Chart rendering state handlers
function handleChartLoaded(chartId) {
  const loaderEl = document.getElementById(`loader_${chartId}`);
  const errorEl = document.getElementById(`error_${chartId}`);
  const imgEl = document.getElementById(`chartImg_${chartId}`);

  if (loaderEl) loaderEl.style.display = 'none';
  if (errorEl) errorEl.style.display = 'none';
  if (imgEl) imgEl.style.display = 'block';
}

function handleChartFailed(chartId) {
  const loaderEl = document.getElementById(`loader_${chartId}`);
  const errorEl = document.getElementById(`error_${chartId}`);
  const imgEl = document.getElementById(`chartImg_${chartId}`);

  if (loaderEl) loaderEl.style.display = 'none';
  if (imgEl) imgEl.style.display = 'none';
  if (errorEl) errorEl.style.display = 'flex';
}

function retryChart(chartId) {
  const loaderEl = document.getElementById(`loader_${chartId}`);
  const errorEl = document.getElementById(`error_${chartId}`);
  const imgEl = document.getElementById(`chartImg_${chartId}`);

  if (errorEl) errorEl.style.display = 'none';
  if (imgEl) imgEl.style.display = 'none';
  if (loaderEl) loaderEl.style.display = 'flex';

  if (imgEl) {
    const timestamp = Date.now();
    imgEl.src = `/api/chart/${chartId}?refresh=1&t=${timestamp}`;
  }
}

function reloadAllCharts() {
  CHART_IDS.forEach(chartId => {
    retryChart(chartId);
  });
}

// =============================================================================
// QUESTION SUBMISSION & CHAT ENGINE
// =============================================================================

function handleQuestionSubmit(event) {
  event.preventDefault();
  const inputEl = document.getElementById('questionInput');
  const question = inputEl.value.trim();
  if (!question) return;

  inputEl.value = '';
  inputEl.style.height = 'auto';
  submitUserQuestion(question);
}

async function submitUserQuestion(questionText) {
  // Ensure we are in the Assistant tab when asking
  switchTab('assistant');

  const heroIntro = document.getElementById('heroIntro');
  const sendBtn = document.getElementById('sendBtn');

  // Hide initial welcome hero on first question
  if (heroIntro) {
    heroIntro.style.display = 'none';
  }

  // 1. Render User Message
  appendUserMessage(questionText);

  // 2. Render Thinking State
  const thinkingId = appendThinkingState();
  if (sendBtn) sendBtn.disabled = true;

  try {
    const response = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: questionText,
        history: conversationHistory
      })
    });

    const data = await response.json();
    removeThinkingState(thinkingId);

    if (!data.success) {
      appendErrorMessage(data.error || "Unable to complete forecast request.");
      return;
    }

    // 3. Render Assistant Response
    appendAssistantResponse(data);

    // 4. Update Conversation Memory
    conversationHistory.push({ role: 'user', content: questionText });
    conversationHistory.push({ role: 'assistant', content: data.answer });

  } catch (err) {
    removeThinkingState(thinkingId);
    appendErrorMessage("Communication error. Please ensure the CarePulse backend server is active.");
    console.error(err);
  } finally {
    if (sendBtn) sendBtn.disabled = false;
  }
}

// =============================================================================
// DOM RENDERING HELPERS
// =============================================================================

function appendUserMessage(text) {
  const stream = document.getElementById('messageStream');
  const div = document.createElement('div');
  div.className = 'msg-user';
  div.textContent = text;
  stream.appendChild(div);
  scrollToBottom();
}

function appendThinkingState() {
  const stream = document.getElementById('messageStream');
  const id = `thinking-${Date.now()}`;
  const div = document.createElement('div');
  div.className = 'thinking-card';
  div.id = id;
  div.innerHTML = `
    <span class="thinking-pulse"></span>
    <span>Analyzing inpatient records and generating forecast...</span>
  `;
  stream.appendChild(div);
  scrollToBottom();
  return id;
}

function removeThinkingState(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function appendErrorMessage(errorText) {
  const stream = document.getElementById('messageStream');
  const div = document.createElement('div');
  div.className = 'msg-assistant';
  div.innerHTML = `
    <div class="assistant-head">
      <span class="assistant-tag" style="color:var(--alert-red);">System Notice</span>
    </div>
    <div class="answer-body">
      <p style="color:var(--alert-red);">${escapeHtml(errorText)}</p>
    </div>
  `;
  stream.appendChild(div);
  scrollToBottom();
}

function appendAssistantResponse(data) {
  const stream = document.getElementById('messageStream');
  const card = document.createElement('div');
  card.className = 'msg-assistant';

  // Format body markdown text
  const formattedHtml = parseSimpleMarkdown(data.answer);

  // Format metrics callout if numerical predictions are returned
  let metricsHtml = '';
  if (data.metrics && data.metrics.predicted_occupancy_rate !== undefined) {
    const m = data.metrics;
    const rateColor = m.is_peak_demand ? 'red' : 'violet';
    const availColor = m.is_peak_demand ? 'red' : 'emerald';

    metricsHtml = `
      <div class="metrics-callout">
        <div class="metric-item">
          <span class="metric-label">Department</span>
          <strong class="metric-val">${escapeHtml(m.department)}</strong>
        </div>
        <div class="metric-item">
          <span class="metric-label">Predicted Occupancy</span>
          <strong class="metric-val ${rateColor}">${m.predicted_occupancy_rate}%</strong>
        </div>
        <div class="metric-item">
          <span class="metric-label">Projected Occupied</span>
          <strong class="metric-val">${m.predicted_occupied_beds} beds</strong>
        </div>
        <div class="metric-item">
          <span class="metric-label">Estimated Available</span>
          <strong class="metric-val ${availColor}">${m.estimated_available_beds} beds</strong>
        </div>
        <div class="metric-item">
          <span class="metric-label">Capacity Buffer</span>
          <strong class="metric-val">${m.total_bed_capacity} total</strong>
        </div>
      </div>
    `;
  }

  // Format follow-up chips
  let followupHtml = '';
  if (data.suggestions && data.suggestions.length > 0) {
    const chipsHtml = data.suggestions.map(s => `
      <button class="followup-btn" onclick="submitUserQuestion('${escapeHtml(s)}')">
        ${escapeHtml(s)}
      </button>
    `).join('');

    followupHtml = `
      <div class="followup-box">
        <span class="followup-title">Recommended Inquiries</span>
        <div class="followup-chips">
          ${chipsHtml}
        </div>
      </div>
    `;
  }

  card.innerHTML = `
    <div class="assistant-head">
      <span class="assistant-tag">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
        CarePulse Research Assistant
      </span>
    </div>
    <div class="answer-body">
      ${formattedHtml}
    </div>
    ${metricsHtml}
    ${followupHtml}
  `;

  stream.appendChild(card);
  scrollToBottom();
}

function resetConversation() {
  conversationHistory = [];
  const stream = document.getElementById('messageStream');
  const heroIntro = document.getElementById('heroIntro');
  if (stream) stream.innerHTML = '';
  if (heroIntro) heroIntro.style.display = 'flex';
  const inputEl = document.getElementById('questionInput');
  if (inputEl) {
    inputEl.value = '';
    inputEl.focus();
  }
}

function scrollToBottom() {
  const viewport = document.getElementById('conversationViewport');
  if (viewport) {
    viewport.scrollTop = viewport.scrollHeight;
  }
}

// =============================================================================
// FORMATTING & SECURITY HELPERS
// =============================================================================

function escapeHtml(text) {
  if (typeof text !== 'string') return '';
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function parseSimpleMarkdown(md) {
  if (!md) return '';

  let html = md;

  // Headings
  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^#### (.*$)/gim, '<h4>$1</h4>');

  // Bold
  html = html.replace(/\*\*(.*?)\*\*/gim, 'strong$1/strong');
  html = html.replace(/strong(.*?)(\/strong)/gim, '<strong>$1</strong>');

  // Inline code
  html = html.replace(/`([^`]+)`/gim, '<code>$1</code>');

  // Unordered lists
  html = html.replace(/^\* (.*$)/gim, '<li>$1</li>');
  html = html.replace(/^- (.*$)/gim, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/gim, '<ul>$1</ul>');
  html = html.replace(/<\/ul>\s*<ul>/gim, '');

  // Paragraph line breaks
  html = html.split('\n\n').map(p => {
    p = p.trim();
    if (!p) return '';
    if (p.startsWith('<h3>') || p.startsWith('<h4>') || p.startsWith('<ul>')) {
      return p;
    }
    return `<p>${p.replace(/\n/g, '<br>')}</p>`;
  }).join('');

  return html;
}

// =============================================================================
// INTELLIGENT BED OCCUPANCY ALERT SYSTEM CLIENT CONTROLLERS
// =============================================================================

let activeAlertsCache = [];

function dismissRedBanner() {
  const banner = document.getElementById('redAlertBanner');
  if (banner) banner.style.display = 'none';
  sessionStorage.setItem('redBannerDismissed', 'true');
}

async function checkAlertsBanner() {
  try {
    const res = await fetch('/api/alerts');
    const data = await res.json();
    if (!data.success) return;

    activeAlertsCache = data.active_alerts || [];
    const count = activeAlertsCache.length;

    // Update sidebar badge
    const badge = document.getElementById('sidebarAlertBadge');
    if (badge) {
      if (count > 0) {
        badge.textContent = count;
        badge.style.display = 'inline-block';
      } else {
        badge.style.display = 'none';
      }
    }

    // Update top red alert banner
    const banner = document.getElementById('redAlertBanner');
    const bannerText = document.getElementById('bannerAlertText');
    const bannerCount = document.getElementById('bannerAlertCount');
    const isDismissed = sessionStorage.getItem('redBannerDismissed') === 'true';

    if (banner && bannerText && bannerCount) {
      if (count > 0 && !isDismissed) {
        const topAlert = activeAlertsCache[0];
        bannerText.textContent = `Hospital occupancy in ${topAlert.department} has reached ${topAlert.occupancy_rate}% (RED ZONE). Immediate capacity assessment and resource planning are recommended.`;
        bannerCount.textContent = count;
        banner.style.display = 'flex';
      } else if (count === 0) {
        banner.style.display = 'none';
      }
    }
  } catch (err) {
    console.warn('Alerts check error:', err);
  }
}

async function fetchAlerts(manual = false) {
  try {
    const res = await fetch('/api/alerts');
    const data = await res.json();
    if (!data.success) return;

    const thresholds = data.thresholds || { green_max: 70.0, red_min: 85.0 };
    const hSummary = data.hospital_summary || {};
    const departments = data.departments || [];
    const activeAlerts = data.active_alerts || [];
    const historyAlerts = data.alerts_history || [];

    // 1. Update Threshold Inputs & Labels
    const lblGreen = document.getElementById('lblGreenRule');
    const lblYellow = document.getElementById('lblYellowRule');
    const lblRed = document.getElementById('lblRedRule');
    const inputGreen = document.getElementById('inputGreenMax');
    const inputRed = document.getElementById('inputRedMin');

    if (lblGreen) lblGreen.textContent = `< ${thresholds.green_max}%`;
    if (lblYellow) lblYellow.textContent = `${thresholds.green_max}% – ${thresholds.red_min}%`;
    if (lblRed) lblRed.textContent = `≥ ${thresholds.red_min}%`;
    if (inputGreen && !inputGreen.matches(':focus')) inputGreen.value = thresholds.green_max;
    if (inputRed && !inputRed.matches(':focus')) inputRed.value = thresholds.red_min;

    // 2. Update Hospital-Wide Census Overview Card
    const elRate = document.getElementById('hospitalOccupancyRate');
    const elZoneTag = document.getElementById('hospitalZoneTag');
    const elZoneLbl = document.getElementById('hospitalZoneLabel');
    const elTot = document.getElementById('hospitalTotalBeds');
    const elOcc = document.getElementById('hospitalOccupiedBeds');
    const elAvail = document.getElementById('hospitalAvailableBeds');
    const elActiveCount = document.getElementById('hospitalActiveAlertsCount');

    if (elRate) elRate.textContent = `${hSummary.occupancy_rate || 0}%`;
    if (elTot) elTot.textContent = hSummary.total_beds || '--';
    if (elOcc) elOcc.textContent = hSummary.occupied_beds || '--';
    if (elAvail) elAvail.textContent = hSummary.available_beds || '--';
    if (elActiveCount) elActiveCount.textContent = activeAlerts.length;

    if (elZoneTag && elZoneLbl) {
      const hZone = (hSummary.zone_info && hSummary.zone_info.zone) || 'RED';
      elZoneTag.className = `census-zone-tag ${hZone.toLowerCase()}`;
      elZoneLbl.textContent = (hSummary.zone_info && hSummary.zone_info.label) || `${hZone} Zone`;
    }

    // 3. Render Active Alerts Stream
    const container = document.getElementById('activeAlertsContainer');
    const counterBadge = document.getElementById('activeAlertsCounter');
    if (counterBadge) counterBadge.textContent = `${activeAlerts.length} Active`;

    if (container) {
      if (activeAlerts.length === 0) {
        container.innerHTML = `
          <div class="alerts-empty-state">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
              <polyline points="22 4 12 14.01 9 11.01"></polyline>
            </svg>
            <h4>All Departments Within Safe Capacity Buffers</h4>
            <p>No hospital departments have reached or are predicted to breach the configured Red Zone threshold (≥ ${thresholds.red_min}%).</p>
          </div>
        `;
      } else {
        container.innerHTML = activeAlerts.map(alert => {
          const isPred = alert.trigger_type === 'PREDICTED_OCCUPANCY';
          const typeClass = isPred ? 'predicted' : 'current';
          const typeLabel = isPred ? 'Predicted Red-Zone Warning' : 'Current Red-Zone Alert';

          return `
            <div class="alert-card-item ${typeClass}" id="card_${alert.id}">
              <div class="alert-card-main">
                <div class="alert-card-badges">
                  <span class="type-pill ${typeClass}">${typeLabel}</span>
                  <span class="alert-dept-name">${alert.department}</span>
                  <span class="alert-timestamp">${alert.timestamp}</span>
                </div>
                <p class="alert-message-text">${alert.message}</p>
                <div class="alert-metrics-strip">
                  <span class="alert-metric-item">Occupancy Rate: <strong>${alert.occupancy_rate}%</strong></span>
                  <span class="alert-metric-item">Available Beds: <strong>${alert.available_beds} beds</strong> (out of ${alert.total_beds})</span>
                  <span class="alert-metric-item">Trigger Context: <em>${alert.explanation}</em></span>
                </div>
              </div>
              <div class="alert-card-action">
                <button class="btn-ack" onclick="acknowledgeAlert('${alert.id}')" title="Acknowledge critical alert">
                  <span>Acknowledge</span>
                </button>
              </div>
            </div>
          `;
        }).join('');
      }
    }

    // 4. Render Department Surveillance Table
    const tbodyDept = document.getElementById('departmentSurveillanceBody');
    if (tbodyDept && departments.length > 0) {
      tbodyDept.innerHTML = departments.map(d => {
        const curZone = (d.zone_info && d.zone_info.zone) || 'UNKNOWN';
        const curZoneClass = curZone.toLowerCase();
        const predZone = (d.forecast_info && d.forecast_info.zone) || 'UNKNOWN';
        const predZoneClass = predZone.toLowerCase();
        const predRate = d.forecast_info ? `${d.forecast_info.predicted_occupancy_rate}%` : 'N/A';

        return `
          <tr>
            <td><strong>${d.department}</strong></td>
            <td>${d.total_beds}</td>
            <td>${d.occupied_beds}</td>
            <td><strong style="color: ${d.available_beds > 0 ? '#10b981' : '#ef4444'}">${d.available_beds}</strong></td>
            <td><strong>${d.occupancy_rate}%</strong></td>
            <td><span class="table-zone-badge ${curZoneClass}">${curZone} ZONE</span></td>
            <td><span class="table-zone-badge ${predZoneClass}">${predZone} (${predRate})</span></td>
          </tr>
        `;
      }).join('');
    }

    // 5. Render Alerts History Table
    const tbodyHist = document.getElementById('alertsHistoryBody');
    if (tbodyHist) {
      if (historyAlerts.length === 0) {
        tbodyHist.innerHTML = `<tr><td colspan="8" class="loading-cell">No prior alerts recorded.</td></tr>`;
      } else {
        tbodyHist.innerHTML = historyAlerts.slice(0, 15).map(a => {
          const isPred = a.trigger_type === 'PREDICTED_OCCUPANCY';
          const typeLabel = isPred ? 'Predicted Warning' : 'Current Critical';
          const statusClass = (a.status || 'ACTIVE').toLowerCase();
          const ackBy = a.acknowledged_by ? `${a.acknowledged_by} (${a.acknowledged_at})` : '-';
          const canAck = a.status === 'ACTIVE';

          return `
            <tr>
              <td><code>${a.id}</code></td>
              <td><strong>${a.department}</strong></td>
              <td>${typeLabel}</td>
              <td>${a.occupancy_rate}%</td>
              <td>${a.available_beds}</td>
              <td>${a.timestamp}</td>
              <td><span class="status-tag-sm ${statusClass}">${a.status}</span></td>
              <td>
                ${canAck ? `<button class="btn-ack" onclick="acknowledgeAlert('${a.id}')">Acknowledge</button>` : `<span style="color: var(--text-dim); font-size: 11.5px;">${ackBy}</span>`}
              </td>
            </tr>
          `;
        }).join('');
      }
    }

    // Synchronize top banner & sidebar badge
    checkAlertsBanner();

  } catch (err) {
    console.error('Failed to fetch alerts:', err);
  }
}

async function acknowledgeAlert(alertId) {
  try {
    const res = await fetch('/api/alerts/acknowledge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ alert_id: alertId, user: 'Clinical Supervisor' })
    });
    const data = await res.json();
    if (data.success) {
      fetchAlerts();
    } else {
      alert(`Could not acknowledge alert: ${data.error}`);
    }
  } catch (err) {
    console.error('Acknowledgment error:', err);
  }
}

async function handleUpdateThresholds(event) {
  event.preventDefault();
  const greenMax = parseFloat(document.getElementById('inputGreenMax').value);
  const redMin = parseFloat(document.getElementById('inputRedMin').value);

  if (isNaN(greenMax) || isNaN(redMin) || greenMax >= redMin || greenMax <= 0 || redMin > 100) {
    alert('Invalid thresholds: Green Max must be strictly less than Red Min, and both must be between 1 and 100.');
    return;
  }

  const btn = document.getElementById('btnSaveThresholds');
  if (btn) btn.textContent = 'Saving...';

  try {
    const res = await fetch('/api/alerts/thresholds', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ green_max: greenMax, red_min: redMin })
    });
    const data = await res.json();
    if (data.success) {
      if (btn) btn.textContent = 'Updated!';
      setTimeout(() => { if (btn) btn.textContent = 'Update Thresholds'; }, 1500);
      fetchAlerts(true);
    } else {
      alert(`Error updating thresholds: ${data.error}`);
      if (btn) btn.textContent = 'Update Thresholds';
    }
  } catch (err) {
    console.error('Threshold update error:', err);
    if (btn) btn.textContent = 'Update Thresholds';
  }
}