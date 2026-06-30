// DualGAT Stock Predictor — frontend logic (v0.4 model switching)
let returnsChart = null;
let currentModel = 'baseline';
let compareMode = false;

const MODEL_COLORS = {
    baseline: '#6c757d',
    ms_lstm:  '#0d6efd',
    dualgat:  '#fd7e14',
    ensemble: '#198754',
};

document.addEventListener('DOMContentLoaded', () => {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    document.getElementById('date-picker').value = yesterday.toISOString().split('T')[0];
    loadModels();
    loadPredictions();
    loadExperts();
    loadBacktest();
    loadSystemStatus();
});

// ── Model management ──

async function loadModels() {
    try {
        const resp = await fetch('/api/models');
        const data = await resp.json();
        window._models = data.models;
        updateModelTabs(data.models);
    } catch (e) {
        console.error('Failed to load models:', e);
    }
}

function updateModelTabs(models) {
    for (const m of models) {
        const tab = document.querySelector(`.model-tab[data-model="${m.id}"]`);
        if (!tab) continue;
        if (!m.available) {
            tab.classList.add('unavailable');
            tab.title = 'Model file not found';
        } else {
            tab.classList.remove('unavailable');
        }
    }
}

function selectModel(modelId) {
    const models = window._models || [];
    const m = models.find(x => x.id === modelId);
    if (m && !m.available) return;

    currentModel = modelId;
    document.querySelectorAll('.model-tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`.model-tab[data-model="${modelId}"]`)?.classList.add('active');
    loadPredictions();
    loadBacktest();
}

function toggleCompare() {
    compareMode = document.getElementById('compare-mode').checked;
    loadPredictions();
}

// ── Utility ──

function setDate(offset) {
    const d = new Date();
    d.setDate(d.getDate() + offset);
    document.getElementById('date-picker').value = d.toISOString().split('T')[0];
    loadPredictions();
    loadExperts();
}

function getDate() {
    return document.getElementById('date-picker').value;
}

function setStatus(msg) {
    document.getElementById('status').textContent = msg;
}

function fmtPct(v) { return (v * 100).toFixed(2) + '%'; }
function fmtPct1(v) { return (v * 100).toFixed(1) + '%'; }

// ── Predictions (updated for model switching) ──

async function loadPredictions() {
    const date = getDate();
    setStatus(`Loading predictions (${currentModel})...`);

    if (compareMode) {
        await loadPredictionsCompare(date);
        return;
    }

    document.getElementById('predictions-compare').style.display = 'none';
    document.getElementById('predictions-table').style.display = '';

    try {
        const resp = await fetch(`/api/predictions?date=${date}&model=${currentModel}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        const experts = data.predictions.filter(p => p.signal_source === 'expert').length;
        const momentum = data.predictions.length - experts;
        document.getElementById('pred-date').textContent =
            `${date} [${currentModel}] (experts: ${experts} | momentum: ${momentum})`;

        if (!data.predictions || data.predictions.length === 0) {
            document.getElementById('predictions-table').innerHTML =
                '<div class="empty-state"><div class="icon">📭</div>No predictions available</div>';
            document.getElementById('pred-summary').innerHTML = '';
            return;
        }

        document.getElementById('pred-summary').innerHTML =
            `<span>🟢 Expert signals: <strong>${experts}</strong> stocks</span>` +
            `<span>⚪ Momentum signals: <strong>${momentum}</strong> stocks</span>` +
            `<span>📊 Coverage: <strong>${data.expert_coverage || '—'}</strong></span>`;

        const cols = [
            { key: 'stock', label: 'Stock' },
            { key: 'predicted_return', label: 'Predicted Return', fmt: v => `<span class="${v > 0 ? 'green' : 'red'}">${fmtPct(v)}</span>` },
            { key: 'signal_source', label: 'Source', fmt: v => {
                if (v === 'expert') return '<span class="badge badge-expert-signal">🧠 Expert</span>';
                if (v === 'ensemble') return '<span class="badge badge-expert-signal">🔗 Ensemble</span>';
                if (v === 'ms_lstm') return '<span class="badge badge-expert-signal">🔮 MS-LSTM</span>';
                if (v === 'dualgat') return '<span class="badge badge-expert-signal">🕸️ DualGAT</span>';
                return '<span class="badge badge-momentum">📐 Momentum</span>';
            }},
        ];
        document.getElementById('predictions-table').innerHTML = buildTable(data.predictions, cols, 20);
        setStatus('Predictions loaded');
    } catch (e) {
        document.getElementById('predictions-table').innerHTML =
            `<div class="empty-state"><div class="icon">⚠️</div>Load failed: ${e.message}</div>`;
        setStatus('Error loading predictions');
    }
}

async function loadPredictionsCompare(date) {
    document.getElementById('predictions-table').style.display = 'none';
    const container = document.getElementById('predictions-compare');
    container.style.display = 'grid';
    container.innerHTML = '<div class="loading">Loading all models...</div>';

    const modelIds = ['baseline', 'ms_lstm', 'dualgat', 'ensemble'];
    const names = { baseline: 'Baseline', ms_lstm: 'MS-LSTM', dualgat: 'DualGAT', ensemble: 'Ensemble' };

    let html = '';
    for (const mid of modelIds) {
        try {
            const resp = await fetch(`/api/predictions?date=${date}&model=${mid}`);
            if (!resp.ok) {
                html += `<div class="mini-card"><h4 style="color:${MODEL_COLORS[mid]};">${names[mid]}</h4><div class="empty-state" style="padding:10px;">⚠️ Unavailable</div></div>`;
                continue;
            }
            const data = await resp.json();
            const rows = data.predictions.slice(0, 10);
            let table = '<table><thead><tr><th>Stock</th><th>Return</th></tr></thead><tbody>';
            for (const p of rows) {
                table += `<tr><td><strong>${p.stock}</strong></td><td class="${p.predicted_return > 0 ? 'green' : 'red'}">${fmtPct(p.predicted_return)}</td></tr>`;
            }
            table += '</tbody></table>';
            html += `<div class="mini-card"><h4 style="color:${MODEL_COLORS[mid]};">${names[mid]}</h4>${table}</div>`;
        } catch (e) {
            html += `<div class="mini-card"><h4 style="color:${MODEL_COLORS[mid]};">${names[mid]}</h4><div class="empty-state">Error</div></div>`;
        }
    }
    container.innerHTML = html;
}

// ── Experts ──

async function loadExperts() {
    const date = getDate();
    setStatus('Loading experts...');
    document.getElementById('expert-date').textContent = date;
    try {
        const resp = await fetch(`/api/experts?date=${date}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        document.getElementById('expert-date').textContent =
            `${date} (experts: ${data.expert_count} | inverse: ${data.inverse_expert_count})`;

        document.getElementById('expert-summary').innerHTML =
            `<span>🟢 Bullish experts: <strong>${data.expert_count}</strong></span>` +
            `<span>🔴 Inverse experts: <strong>${data.inverse_expert_count}</strong></span>` +
            `<span>📋 Total: <strong>${data.total}</strong> signals</span>`;

        if (!data.experts || data.experts.length === 0) {
            document.getElementById('experts-table').innerHTML =
                '<div class="empty-state"><div class="icon">🔍</div>No expert signals today<br><small>Possible reasons: insufficient data or no users meeting thresholds</small></div>';
            return;
        }

        const typeLabel = t => t === 'expert' ? '🟢 Expert' : '🔴 Inverse';
        const typeClass = t => t === 'expert' ? 'badge-expert' : 'badge-inverse';
        const dirClass = d => d === 'Bullish' ? 'badge-bullish' : 'badge-bearish';
        const dirLabel = d => d === 'Bullish' ? '📈 Bullish' : '📉 Bearish';

        const cols = [
            { key: 'user_id', label: 'User' },
            { key: 'stock', label: 'Stock', fmt: v => `<strong>${v}</strong>` },
            { key: 'expert_type', label: 'Type', fmt: v => `<span class="badge ${typeClass(v)}">${typeLabel(v)}</span>` },
            { key: 'predicted_direction', label: 'Direction', fmt: v => `<span class="badge ${dirClass(v)}">${dirLabel(v)}</span>` },
            { key: 'accuracy_recent', label: 'Recent Acc', fmt: v => fmtPct1(v) },
            { key: 'accuracy_long', label: 'Long Acc', fmt: v => fmtPct1(v) },
        ];
        document.getElementById('experts-table').innerHTML = buildTable(data.experts, cols, 30);
        setStatus('Experts loaded');
    } catch (e) {
        document.getElementById('experts-table').innerHTML =
            `<div class="empty-state"><div class="icon">⚠️</div>Load failed: ${e.message}</div>`;
        setStatus('Error loading experts');
    }
}

// ── Backtest (updated — multi-line chart + comparison table) ──

async function loadBacktest() {
    const endDate = getDate();
    const startDate = new Date(endDate);
    startDate.setDate(startDate.getDate() - 90);
    const start = startDate.toISOString().split('T')[0];

    try {
        const resp = await fetch(`/api/backtest/compare?start=${start}&end=${endDate}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        const modelIds = Object.keys(data.models);
        if (modelIds.length === 0) {
            document.getElementById('chart-container').innerHTML =
                '<div class="empty-state"><div class="icon">📉</div>No backtest data available</div>';
            return;
        }

        // Show current model's metrics in the summary bar
        const cur = data.models[currentModel] || data.models[modelIds[0]];
        const arClass = cur.annualized_return > 0 ? 'green' : 'red';
        document.querySelector('#metrics-summary').innerHTML = `
            <div class="metric"><div class="value ${arClass}">${fmtPct1(cur.annualized_return)}</div><div class="label">Annualized Return</div></div>
            <div class="metric"><div class="value">${cur.sharpe_ratio.toFixed(2)}</div><div class="label">Sharpe Ratio</div></div>
            <div class="metric"><div class="value">${fmtPct(cur.mean_ic)}</div><div class="label">Mean IC</div></div>
            <div class="metric"><div class="value red">${fmtPct1(cur.max_drawdown)}</div><div class="label">Max Drawdown</div></div>
        `;

        // Multi-line chart
        if (returnsChart) returnsChart.destroy();
        const ctx = document.getElementById('returnsChart').getContext('2d');
        const maxLen = Math.max(...modelIds.map(id => (data.models[id].cumulative_returns || []).length));

        const datasets = modelIds.map(id => {
            const cr = data.models[id].cumulative_returns || [];
            return {
                label: id.charAt(0).toUpperCase() + id.slice(1),
                data: cr,
                borderColor: MODEL_COLORS[id] || '#999',
                backgroundColor: 'transparent',
                tension: 0.3,
                pointRadius: 0,
                borderWidth: id === currentModel ? 3 : 1.5,
            };
        });

        returnsChart = new Chart(ctx, {
            type: 'line',
            data: { labels: Array.from({length: maxLen}, (_, i) => i + 1), datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                plugins: {
                    legend: { display: true, position: 'bottom', labels: { boxWidth: 12, font: { size: 10 } } },
                },
                scales: {
                    x: { display: true, title: { display: true, text: 'Trading Day', color: '#888' }, ticks: { maxTicksLimit: 15, color: '#aaa' }, grid: { display: false } },
                    y: { ticks: { callback: v => (v * 100).toFixed(0) + '%', color: '#aaa' }, grid: { color: '#f0f0f0' } },
                }
            }
        });

        // Comparison table
        const names = { baseline: 'Baseline', ms_lstm: 'MS-LSTM', dualgat: 'DualGAT', ensemble: 'Ensemble' };
        const metrics = ['annualized_return', 'sharpe_ratio', 'mean_ic', 'max_drawdown', 'icir'];
        const labels = ['Ann. Return', 'Sharpe', 'Mean IC', 'Max DD', 'ICIR'];
        const fmt = [
            v => fmtPct1(v),
            v => v.toFixed(2),
            v => fmtPct(v),
            v => fmtPct1(v),
            v => v.toFixed(2),
        ];
        const higherBetter = [true, true, true, false, true];

        // Find best values
        const best = {};
        for (let i = 0; i < metrics.length; i++) {
            const vals = modelIds.map(id => data.models[id][metrics[i]]).filter(v => v != null);
            if (vals.length > 0) {
                best[metrics[i]] = higherBetter[i] ? Math.max(...vals) : Math.min(...vals);
            }
        }

        let tbl = '<table class="compare-table"><thead><tr><th>Model</th>';
        for (const l of labels) tbl += `<th>${l}</th>`;
        tbl += '</tr></thead><tbody>';
        for (const id of modelIds) {
            const m = data.models[id];
            tbl += `<tr><td><strong>${names[id] || id}</strong></td>`;
            for (let i = 0; i < metrics.length; i++) {
                const val = m[metrics[i]];
                const isBest = best[metrics[i]] !== undefined && val === best[metrics[i]];
                tbl += `<td class="${isBest ? 'best' : ''}">${val != null ? fmt[i](val) : '—'}</td>`;
            }
            tbl += '</tr>';
        }
        tbl += '</tbody></table>';
        document.getElementById('compare-table-container').innerHTML = tbl;

        const totalDays = modelIds.reduce((max, id) => Math.max(max, data.models[id].n_trading_days || 0), 0);
        document.getElementById('chart-legend').textContent =
            `${totalDays} trading days · ${start} ~ ${endDate}`;
    } catch (e) {
        console.error('Backtest load error:', e);
        document.getElementById('chart-container').innerHTML =
            `<div class="empty-state"><div class="icon">⚠️</div>Backtest load failed: ${e.message}</div>`;
    }
}

// ── System Status ──

async function loadSystemStatus() {
    try {
        const stocksResp = await fetch('/api/stocks');
        const stocksData = await stocksResp.json();

        const date = getDate();
        const expertsResp = await fetch(`/api/experts?date=${date}`);
        const expertsData = await expertsResp.json();

        const predsResp = await fetch(`/api/predictions?date=${date}&model=${currentModel}`);
        const predsData = await predsResp.json();
        if (!predsData.predictions) return;
        const expertPreds = predsData.predictions.filter(p => p.signal_source === 'expert').length;

        document.getElementById('sys-stocks').textContent = stocksData.count;
        document.getElementById('sys-experts').textContent = expertsData.total;
        document.getElementById('sys-coverage').textContent = expertPreds + '/' + stocksData.count;
        document.getElementById('sys-posts').textContent = '—';
    } catch (e) {
        // System status is non-critical
    }
}

// ── Data Collection ──

async function triggerCollect() {
    setStatus('Collecting data...');
    const endDate = getDate();
    const startDate = new Date(endDate);
    startDate.setDate(startDate.getDate() - 3);
    const start = startDate.toISOString().split('T')[0];

    const es = new EventSource(
        `/api/collect/stream?start=${start}&end=${endDate}`
    );

    es.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setStatus(data.msg);
        if (data.step === 'done') {
            es.close();
            loadPredictions();
            loadExperts();
            loadBacktest();
            loadSystemStatus();
        }
    };

    es.onerror = () => {
        es.close();
        setStatus('Collection connection error, please retry');
    };
}

// ── Table builder utility ──

function buildTable(rows, cols, maxRows) {
    let html = '<table><thead><tr>';
    for (const c of cols) html += `<th>${c.label}</th>`;
    html += '</tr></thead><tbody>';
    const shown = rows.slice(0, maxRows);
    for (const row of shown) {
        html += '<tr>';
        for (const c of cols) {
            const raw = row[c.key];
            html += '<td>' + (c.fmt ? c.fmt(raw) : raw) + '</td>';
        }
        html += '</tr>';
    }
    html += '</tbody></table>';
    if (rows.length > maxRows) {
        html += `<div style="text-align:center;padding:8px;color:#888;font-size:12px;">Showing ${maxRows} of ${rows.length} rows</div>`;
    }
    return html;
}

// ── Auto-refresh every 5 minutes ──
setInterval(() => {
    loadPredictions();
    loadExperts();
    loadBacktest();
    loadSystemStatus();
}, 300000);
