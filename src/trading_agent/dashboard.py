from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from trading_agent.audit import AuditStore


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, server_address, store: AuditStore) -> None:
        self.store = store
        super().__init__(server_address, DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(DASHBOARD_HTML)
            return
        if parsed.path == "/api/summary":
            self._send_json(self.server.store.latest_summary())
            return
        if parsed.path == "/api/cycles":
            params = parse_qs(parsed.query)
            limit = self._int_param(params, "limit", 25)
            self._send_json({"cycles": self.server.store.recent_cycles(limit=limit)})
            return
        if parsed.path == "/api/events":
            params = parse_qs(parsed.query)
            limit = self._int_param(params, "limit", 200)
            cycle_id = self._optional_int_param(params, "cycle_id")
            event_type = params.get("event_type", [None])[0] or None
            self._send_json(
                {
                    "events": self.server.store.events(
                        cycle_id=cycle_id,
                        event_type=event_type,
                        limit=limit,
                    )
                }
            )
            return
        if parsed.path == "/api/performance":
            self._send_json(self.server.store.performance_report())
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args) -> None:
        return

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _int_param(self, params: dict[str, list[str]], key: str, default: int) -> int:
        try:
            return int(params.get(key, [default])[0])
        except (TypeError, ValueError):
            return default

    def _optional_int_param(self, params: dict[str, list[str]], key: str) -> int | None:
        value = params.get(key, [None])[0]
        if value in {None, ""}:
            return None
        try:
            return int(value)
        except ValueError:
            return None


def serve_dashboard(database_path: str | Path, host: str = "127.0.0.1", port: int = 8080) -> None:
    store = AuditStore(database_path)
    server = DashboardServer((host, port), store)
    print(f"Dashboard running at http://{host}:{port}")
    server.serve_forever()


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trading Agent Dashboard</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --surface: #ffffff;
      --text: #20242a;
      --muted: #68717d;
      --line: #d9dee5;
      --accent: #0f766e;
      --danger: #b42318;
      --warn: #b54708;
      --ok: #067647;
      --shadow: 0 1px 2px rgba(16, 24, 40, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      background: var(--surface);
      border-bottom: 1px solid var(--line);
      padding: 14px 20px;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .topbar {
      align-items: center;
      display: flex;
      gap: 18px;
      justify-content: space-between;
      max-width: 1480px;
      margin: 0 auto;
    }
    h1 { font-size: 18px; margin: 0; font-weight: 700; }
    .meta { color: var(--muted); font-size: 13px; }
    main { max-width: 1480px; margin: 0 auto; padding: 18px 20px 36px; }
    .metrics {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(5, minmax(140px, 1fr));
      margin-bottom: 16px;
    }
    .metric {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-height: 78px;
      padding: 12px;
    }
    .metric span { color: var(--muted); display: block; font-size: 12px; }
    .metric strong { display: block; font-size: 21px; margin-top: 6px; white-space: nowrap; }
    .tabs {
      align-items: center;
      border-bottom: 1px solid var(--line);
      display: flex;
      gap: 6px;
      margin-top: 8px;
    }
    .tab {
      background: transparent;
      border: 0;
      border-bottom: 3px solid transparent;
      color: var(--muted);
      cursor: pointer;
      font: inherit;
      padding: 12px 10px 10px;
    }
    .tab.active { border-bottom-color: var(--accent); color: var(--text); font-weight: 700; }
    .toolbar {
      align-items: center;
      display: flex;
      gap: 10px;
      justify-content: space-between;
      margin: 14px 0 10px;
    }
    select, button {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--text);
      font: inherit;
      min-height: 34px;
      padding: 6px 10px;
    }
    button { cursor: pointer; }
    table {
      border-collapse: collapse;
      background: var(--surface);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      width: 100%;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }
    th {
      background: #eef1f4;
      color: #374151;
      font-size: 12px;
      position: sticky;
      top: 57px;
      z-index: 5;
    }
    tbody tr:hover { background: #fafafa; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      display: inline-block;
      font-size: 12px;
      line-height: 1;
      padding: 4px 8px;
      white-space: nowrap;
    }
    .approved, .submitted, .completed { color: var(--ok); border-color: rgba(6, 118, 71, 0.35); }
    .rejected, .failed { color: var(--danger); border-color: rgba(180, 35, 24, 0.35); }
    .running, .generated, .captured { color: var(--warn); border-color: rgba(181, 71, 8, 0.35); }
    .helping { color: var(--ok); border-color: rgba(6, 118, 71, 0.35); }
    .hurting { color: var(--danger); border-color: rgba(180, 35, 24, 0.35); }
    .neutral, .needs_data { color: var(--warn); border-color: rgba(181, 71, 8, 0.35); }
    .hidden { display: none; }
    .empty {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
      padding: 28px;
      text-align: center;
    }
    .detail {
      color: var(--muted);
      max-width: 520px;
      overflow-wrap: anywhere;
    }
    .json {
      background: #111827;
      border-radius: 6px;
      color: #f9fafb;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      max-height: 220px;
      max-width: 620px;
      overflow: auto;
      padding: 10px;
      white-space: pre-wrap;
    }
    @media (max-width: 900px) {
      .topbar, .toolbar { align-items: flex-start; flex-direction: column; }
      .metrics { grid-template-columns: repeat(2, minmax(130px, 1fr)); }
      th { position: static; }
      table { display: block; overflow-x: auto; }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div>
        <h1>Trading Agent Dashboard</h1>
        <div class="meta" id="cycleMeta">Waiting for audit data</div>
      </div>
      <div class="meta" id="refreshMeta">Auto-refreshing every 10s</div>
    </div>
  </header>
  <main>
    <section class="metrics" id="metrics"></section>
    <nav class="tabs">
      <button class="tab active" data-tab="decisions">Trade Decisions</button>
      <button class="tab" data-tab="orders">Orders</button>
      <button class="tab" data-tab="performance">Performance</button>
      <button class="tab" data-tab="market">Market</button>
      <button class="tab" data-tab="research">Research</button>
      <button class="tab" data-tab="audit">Audit History</button>
    </nav>
    <section id="decisions" class="panel"></section>
    <section id="orders" class="panel hidden"></section>
    <section id="performance" class="panel hidden"></section>
    <section id="market" class="panel hidden"></section>
    <section id="research" class="panel hidden"></section>
    <section id="audit" class="panel hidden">
      <div class="toolbar">
        <div>
          <select id="eventType">
            <option value="">All events</option>
            <option value="market_snapshot">Market snapshots</option>
            <option value="research_result">Research results</option>
            <option value="trade_candidate">Trade candidates</option>
            <option value="risk_decision">Risk decisions</option>
            <option value="order">Orders</option>
            <option value="broker_order_update">Broker order updates</option>
            <option value="risk_stop">Risk stops</option>
          </select>
          <button id="reloadAudit">Refresh</button>
        </div>
        <div class="meta">Newest events first</div>
      </div>
      <div id="auditTable"></div>
    </section>
  </main>
  <script>
    const state = { summary: null, performance: null, auditEvents: [] };
    const currency = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

    function fmtMoney(value) {
      const num = Number(value || 0);
      return currency.format(num);
    }
    function fmtSignedMoney(value) {
      const num = Number(value || 0);
      return `${num > 0 ? "+" : ""}${currency.format(num)}`;
    }
    function fmtPct(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "";
      return `${Number(value).toFixed(2)}%`;
    }
    function fmtDate(value) {
      if (!value) return "";
      return new Date(value).toLocaleString();
    }
    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[char]));
    }
    function statusPill(value) {
      if (value === null || value === undefined || value === "") return "";
      return `<span class="pill ${esc(value)}">${esc(value)}</span>`;
    }
    function payload(event) {
      return event.payload || {};
    }
    function renderTable(containerId, headers, rows, emptyText) {
      const container = document.getElementById(containerId);
      if (!rows.length) {
        container.innerHTML = `<div class="empty">${esc(emptyText)}</div>`;
        return;
      }
      container.innerHTML = `
        <table>
          <thead><tr>${headers.map(h => `<th class="${h.cls || ""}">${esc(h.label)}</th>`).join("")}</tr></thead>
          <tbody>${rows.map(row => `<tr>${headers.map(h => `<td class="${h.cls || ""}">${row[h.key] ?? ""}</td>`).join("")}</tr>`).join("")}</tbody>
        </table>
      `;
    }
    function renderMetrics(summary) {
      const cycle = summary.cycle;
      const account = cycle?.account || {};
      const decisions = summary.decisions || [];
      const orders = summary.orders || [];
      const approved = decisions.filter(item => item.approved).length;
      const submitted = orders.filter(item => item.status === "submitted").length;
      const rejected = orders.filter(item => item.status === "rejected").length;
      document.getElementById("metrics").innerHTML = [
        ["Equity", fmtMoney(account.equity)],
        ["Cash", fmtMoney(account.cash)],
        ["Buying Power", fmtMoney(account.buying_power)],
        ["Approved Decisions", approved],
        ["Submitted / Rejected", `${submitted} / ${rejected}`],
      ].map(([label, value]) => `<div class="metric"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join("");
      document.getElementById("cycleMeta").textContent = cycle
        ? `Cycle #${cycle.id} • ${cycle.status} • started ${fmtDate(cycle.started_at)}`
        : "No agent cycle has been recorded yet";
    }
    function renderDecisions(summary) {
      const rows = (summary.decisions || []).slice().reverse().map(event => {
        const candidate = payload(event).candidate || {};
        const intent = payload(event).intent || {};
        return {
          approved: statusPill(event.approved ? "approved" : "rejected"),
          symbol: esc(event.symbol),
          strategy: esc(event.strategy),
          side: esc(candidate.side || intent.side || ""),
          score: esc(Number(event.score || 0).toFixed(2)),
          entry: esc(candidate.entry_price ?? ""),
          qty: esc(intent.qty ?? intent.notional ?? ""),
          reason: `<div class="detail">${esc(event.reason)}</div>`,
        };
      });
      renderTable("decisions", [
        { key: "approved", label: "Approved" },
        { key: "symbol", label: "Symbol" },
        { key: "strategy", label: "Strategy" },
        { key: "side", label: "Side" },
        { key: "score", label: "Score", cls: "num" },
        { key: "entry", label: "Entry", cls: "num" },
        { key: "qty", label: "Qty / Notional", cls: "num" },
        { key: "reason", label: "Reason" },
      ], rows, "No risk decisions recorded yet.");
    }
    function renderOrders(summary) {
      const rows = (summary.orders || []).map(event => {
        const data = payload(event);
        const brokerOrder = data.broker_order || {};
        const intent = data.intent || {};
        return {
          status: statusPill(event.status),
          symbol: esc(event.symbol),
          strategy: esc(event.strategy),
          side: esc(intent.side || ""),
          qty: esc(intent.qty ?? intent.notional ?? ""),
          orderId: esc(brokerOrder.id || ""),
          reason: `<div class="detail">${esc(event.reason || data.error || "")}</div>`,
          time: esc(fmtDate(event.created_at)),
        };
      });
      renderTable("orders", [
        { key: "status", label: "Status" },
        { key: "symbol", label: "Symbol" },
        { key: "strategy", label: "Strategy" },
        { key: "side", label: "Side" },
        { key: "qty", label: "Qty / Notional", cls: "num" },
        { key: "orderId", label: "Broker Order ID" },
        { key: "reason", label: "Reason" },
        { key: "time", label: "Time" },
      ], rows, "No submitted or rejected orders recorded yet.");
    }
    function renderPerformance(report) {
      const summary = report.summary || {};
      const warnings = report.data_quality?.warnings || [];
      const metricHtml = [
        ["Total P/L", fmtSignedMoney(summary.total_pl)],
        ["Max Drawdown", fmtPct(summary.max_drawdown_pct)],
        ["Win Rate", fmtPct(summary.win_rate_pct)],
        ["Realized P/L", fmtSignedMoney(summary.realized_pl)],
        ["Open P/L", fmtSignedMoney(summary.open_unrealized_pl)],
      ].map(([label, value]) => `<div class="metric"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join("");

      const strategyRows = (report.strategies || []).map(row => ({
        assessment: statusPill(row.assessment),
        strategy: esc(row.strategy),
        total: esc(fmtSignedMoney(row.total_pl_estimate)),
        realized: esc(fmtSignedMoney(row.realized_pl)),
        open: esc(fmtSignedMoney(row.open_unrealized_pl)),
        exitSignal: esc(fmtSignedMoney(row.exit_signal_pl)),
        winRate: esc(fmtPct(row.win_rate_pct)),
        closed: esc(row.closed_trades),
        exits: esc(row.exit_signals),
        orders: esc(`${row.submitted_orders}/${row.rejected_orders}/${row.skipped_orders}`),
      }));
      const strategyHeaders = [
        { key: "assessment", label: "Helping" },
        { key: "strategy", label: "Strategy" },
        { key: "total", label: "P/L Estimate", cls: "num" },
        { key: "realized", label: "Realized", cls: "num" },
        { key: "open", label: "Open", cls: "num" },
        { key: "exitSignal", label: "Exit Signal", cls: "num" },
        { key: "winRate", label: "Win Rate", cls: "num" },
        { key: "closed", label: "Closed", cls: "num" },
        { key: "exits", label: "Exit Signals", cls: "num" },
        { key: "orders", label: "Submitted / Rejected / Skipped", cls: "num" },
      ];

      const positionRows = (report.open_positions || []).map(row => ({
        symbol: esc(row.symbol),
        strategy: esc(row.strategy),
        assetClass: esc(row.asset_class),
        qty: esc(row.qty),
        marketValue: esc(fmtMoney(row.market_value)),
        open: esc(fmtSignedMoney(row.unrealized_pl)),
      }));
      const recentCurve = (report.equity_curve || []).slice(-12).reverse().map(row => ({
        cycle: esc(row.cycle_id),
        time: esc(fmtDate(row.started_at)),
        equity: esc(fmtMoney(row.equity)),
        drawdown: esc(fmtPct(row.drawdown_pct)),
      }));

      document.getElementById("performance").innerHTML = `
        <section class="metrics">${metricHtml}</section>
        ${warnings.length ? `<div class="empty">${warnings.map(esc).join("<br>")}</div>` : ""}
        <div class="toolbar"><div class="meta">Strategy performance</div><div class="meta">P/L estimate = realized + exit-signal + open P/L</div></div>
        <div id="strategyPerformance"></div>
        <div class="toolbar"><div class="meta">Open positions by strategy</div></div>
        <div id="openPositionPerformance"></div>
        <div class="toolbar"><div class="meta">Recent equity drawdown</div></div>
        <div id="equityPerformance"></div>
      `;
      renderTable("strategyPerformance", strategyHeaders, strategyRows, "No strategy performance data yet.");
      renderTable("openPositionPerformance", [
        { key: "symbol", label: "Symbol" },
        { key: "strategy", label: "Strategy" },
        { key: "assetClass", label: "Class" },
        { key: "qty", label: "Qty", cls: "num" },
        { key: "marketValue", label: "Market Value", cls: "num" },
        { key: "open", label: "Open P/L", cls: "num" },
      ], positionRows, "No open positions recorded in the latest cycle.");
      renderTable("equityPerformance", [
        { key: "cycle", label: "Cycle", cls: "num" },
        { key: "time", label: "Time" },
        { key: "equity", label: "Equity", cls: "num" },
        { key: "drawdown", label: "Drawdown", cls: "num" },
      ], recentCurve, "No equity history recorded yet.");
    }
    function renderMarket(summary) {
      const rows = (summary.market_snapshots || []).map(event => {
        const data = payload(event);
        return {
          symbol: esc(event.symbol),
          assetClass: esc(data.asset_class),
          price: esc(data.price),
          closes: esc((data.closes || []).length),
          asOf: esc(fmtDate(data.as_of)),
        };
      });
      renderTable("market", [
        { key: "symbol", label: "Symbol" },
        { key: "assetClass", label: "Class" },
        { key: "price", label: "Price", cls: "num" },
        { key: "closes", label: "Bars", cls: "num" },
        { key: "asOf", label: "As Of" },
      ], rows, "No market snapshots recorded yet.");
    }
    function renderResearch(summary) {
      const rows = (summary.research_results || []).map(event => {
        const data = payload(event);
        const filings = data.sec_summary?.recent_filings || [];
        const latestFiling = filings[0] ? `${filings[0].form} ${filings[0].filing_date}` : "";
        const regime = data.crypto_summary?.regime;
        const cryptoText = regime ? `${regime.label} ${regime.score}` : "";
        return {
          symbol: esc(event.symbol),
          entity: esc(data.sec_summary?.entity_name || ""),
          crypto: esc(cryptoText),
          headlines: esc((data.news || []).length),
          filing: esc(latestFiling),
          notes: `<div class="detail">${esc((data.notes || []).join(" | "))}</div>`,
        };
      });
      renderTable("research", [
        { key: "symbol", label: "Symbol" },
        { key: "entity", label: "Entity" },
        { key: "crypto", label: "Crypto Regime" },
        { key: "headlines", label: "Headlines", cls: "num" },
        { key: "filing", label: "Latest Filing" },
        { key: "notes", label: "Notes" },
      ], rows, "No research results recorded yet.");
    }
    async function loadSummary() {
      const response = await fetch("/api/summary", { cache: "no-store" });
      state.summary = await response.json();
      renderMetrics(state.summary);
      renderDecisions(state.summary);
      renderOrders(state.summary);
      renderMarket(state.summary);
      renderResearch(state.summary);
      document.getElementById("refreshMeta").textContent = `Updated ${new Date().toLocaleTimeString()} • Auto-refreshing every 10s`;
    }
    async function loadAudit() {
      const eventType = document.getElementById("eventType").value;
      const query = new URLSearchParams({ limit: "300" });
      if (eventType) query.set("event_type", eventType);
      const response = await fetch(`/api/events?${query.toString()}`, { cache: "no-store" });
      const data = await response.json();
      state.auditEvents = data.events || [];
      const rows = state.auditEvents.map(event => ({
        id: esc(event.id),
        cycle: esc(event.cycle_id),
        time: esc(fmtDate(event.created_at)),
        type: esc(event.event_type),
        symbol: esc(event.symbol || ""),
        status: statusPill(event.status || (event.approved === true ? "approved" : event.approved === false ? "rejected" : "")),
        reason: `<div class="detail">${esc(event.reason || "")}</div>`,
        payload: `<pre class="json">${esc(JSON.stringify(event.payload, null, 2))}</pre>`,
      }));
      renderTable("auditTable", [
        { key: "id", label: "ID", cls: "num" },
        { key: "cycle", label: "Cycle", cls: "num" },
        { key: "time", label: "Time" },
        { key: "type", label: "Type" },
        { key: "symbol", label: "Symbol" },
        { key: "status", label: "Status" },
        { key: "reason", label: "Reason" },
        { key: "payload", label: "Payload" },
      ], rows, "No audit events recorded yet.");
    }
    async function loadPerformance() {
      const response = await fetch("/api/performance", { cache: "no-store" });
      state.performance = await response.json();
      renderPerformance(state.performance);
    }
    document.querySelectorAll(".tab").forEach(tab => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(item => item.classList.remove("active"));
        document.querySelectorAll(".panel").forEach(item => item.classList.add("hidden"));
        tab.classList.add("active");
        document.getElementById(tab.dataset.tab).classList.remove("hidden");
        if (tab.dataset.tab === "audit") loadAudit();
        if (tab.dataset.tab === "performance") loadPerformance();
      });
    });
    document.getElementById("reloadAudit").addEventListener("click", loadAudit);
    document.getElementById("eventType").addEventListener("change", loadAudit);
    loadSummary().catch(error => {
      document.getElementById("cycleMeta").textContent = `Dashboard load failed: ${error}`;
    });
    setInterval(() => {
      loadSummary();
      if (!document.getElementById("performance").classList.contains("hidden")) loadPerformance();
      if (!document.getElementById("audit").classList.contains("hidden")) loadAudit();
    }, 10000);
  </script>
</body>
</html>"""
