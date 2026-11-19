function modePillClass(mode) {
  if (mode === "COMPLETED" || mode === "APPROVED") return "pill pill-ok";
  if (mode === "HOLD_REACQUIRE" || mode === "PREEMPTIVE_HOLD") return "pill pill-warn";
  if (mode === "SAFE_HOLD" || mode === "REJECTED") return "pill pill-err";
  return "pill";
}

document.getElementById("btn-fill-violation").addEventListener("click", () => {
  document.getElementById("val-mission-id").value = "MSN-KEEPOUT-TEST";
  document.getElementById("val-sector").value = "KEEPOUT_K2";
  document.getElementById("val-joint-deg").value = "148.0";
  document.getElementById("val-velocity").value = "6.2";
  document.getElementById("val-battery").value = "15";
});

document.getElementById("validator-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const resultBox = document.getElementById("validator-result");
  resultBox.innerHTML = "Evaluating command envelope...";

  const zonesRaw = document.getElementById("val-zones").value;
  const restrictedZones = zonesRaw
    .split(",")
    .map((z) => z.trim())
    .filter(Boolean);

  const payload = {
    mission_id: document.getElementById("val-mission-id").value.trim(),
    target_sector: document.getElementById("val-sector").value.trim(),
    target_joint_deg: parseFloat(document.getElementById("val-joint-deg").value),
    max_joint_velocity_deg_s: parseFloat(document.getElementById("val-velocity").value),
    battery_soc_pct: parseFloat(document.getElementById("val-battery").value),
    restricted_zones: restrictedZones,
  };

  try {
    const res = await fetch("/mission/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (res.ok) {
      resultBox.innerHTML = `
        <p><span class="pill pill-ok">APPROVED</span> Mission <strong>${data.mission_id}</strong></p>
        <p style="margin-top:8px;">Target Sector: <strong>${data.sector}</strong></p>
        <p>Enforced Joint Velocity Limit: <strong>${data.enforced_velocity_limit_deg_s} deg/s</strong></p>
        <p>Safe Envelope Check: <strong>Passed all kinematic, power, and keep-out checks</strong></p>
      `;
    } else {
      const detail = data.detail || {};
      resultBox.innerHTML = `
        <p><span class="pill pill-err">REJECTED (HTTP ${res.status})</span></p>
        <p style="margin-top:8px;">Error Code: <code>${detail.error_code || "VALIDATION_ERROR"}</code></p>
        <p style="margin-top:4px;">Reason: <strong>${detail.message || JSON.stringify(data)}</strong></p>
      `;
    }
  } catch (err) {
    resultBox.innerHTML = `<p>Request failed: ${err.message}</p>`;
  }
});

async function runPreset(preset) {
  const summaryEl = document.getElementById("summary") || document.getElementById("sim-summary");
  const wrapEl = document.getElementById("sim-table-wrap");
  summaryEl.innerHTML = `<span>Executing <strong>${preset}</strong> telemetry stream...</span>`;

  try {
    const res = await fetch(`/simulation/run-preset/${preset}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) {
      wrapEl.innerHTML = `<p class="muted-note">Error: ${JSON.stringify(data)}</p>`;
      return;
    }

    summaryEl.innerHTML = `
      <span>Mission: <strong>${data.validation.mission_id}</strong> (${data.validation.sector})</span>
      <span>Final Mode: <span class="${modePillClass(data.final_mode)}">${data.final_mode}</span></span>
      <span>Ticks Executed: <strong>${data.ticks_executed}</strong></span>
      <span>Latched Fault: <strong>${data.last_fault_code || "None"}</strong></span>
    `;

    const rows = data.history
      .map(
        (h) => `
        <tr>
          <td>#${h.tick}</td>
          <td><span class="${modePillClass(h.mode)}">${h.mode}</span></td>
          <td><code>${h.active_node}</code></td>
          <td>${h.status}</td>
          <td>${(h.vision_confidence * 100).toFixed(0)}%</td>
          <td>${h.link_active ? "UP" : "DOWN"}</td>
          <td>${(h.anomaly_severity * 100).toFixed(0)}% (${h.anomaly_driver})</td>
          <td>${h.reason}</td>
        </tr>
      `
      )
      .join("");

    wrapEl.innerHTML = `
      <table>
        <thead>
          <tr>
            <th>Tick</th>
            <th>Mode</th>
            <th>Active BT Node</th>
            <th>Status</th>
            <th>Vision</th>
            <th>Telemetry Link</th>
            <th>ML Severity</th>
            <th>Decision Rationale</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  } catch (err) {
    wrapEl.innerHTML = `<p class="muted-note">Request failed: ${err.message}</p>`;
  }
}