// DualGAT 股票预测系统 — 前端逻辑 (中文版)
let returnsChart = null;

document.addEventListener('DOMContentLoaded', () => {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    document.getElementById('date-picker').value = yesterday.toISOString().split('T')[0];
    loadPredictions();
    loadExperts();
    loadBacktest();
    loadSystemStatus();
});

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

async function loadPredictions() {
    const date = getDate();
    setStatus('正在加载预测...');
    document.getElementById('pred-date').textContent = date;
    try {
        const resp = await fetch(`/api/predictions?date=${date}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        const experts = data.predictions.filter(p => p.signal_source === 'expert').length;
        const momentum = data.predictions.length - experts;
        document.getElementById('pred-date').textContent =
            `${date}（专家: ${experts} | 动量: ${momentum}）`;

        if (!data.predictions || data.predictions.length === 0) {
            document.getElementById('predictions-table').innerHTML =
                '<div class="empty-state"><div class="icon">📭</div>暂无预测数据</div>';
            document.getElementById('pred-summary').innerHTML = '';
            return;
        }

        // 信号来源分布摘要
        document.getElementById('pred-summary').innerHTML =
            `<span>🟢 专家信号: <strong>${experts}</strong> 只股票</span>` +
            `<span>⚪ 动量信号: <strong>${momentum}</strong> 只股票</span>` +
            `<span>📊 覆盖率: <strong>${data.expert_coverage}</strong></span>`;

        const cols = [
            { key: 'stock', label: '股票' },
            { key: 'predicted_return', label: '预测收益率', fmt: v => `<span class="${v > 0 ? 'green' : 'red'}">${fmtPct(v)}</span>` },
            { key: 'signal_source', label: '信号来源', fmt: v => {
                if (v === 'expert') return '<span class="badge badge-expert-signal">🧠 专家信号</span>';
                if (v === 'momentum') return '<span class="badge badge-momentum">📐 动量因子</span>';
                return v;
            }},
        ];
        document.getElementById('predictions-table').innerHTML = buildTable(data.predictions, cols, 20);
        setStatus('预测加载完成 ✓');
    } catch (e) {
        document.getElementById('predictions-table').innerHTML =
            `<div class="empty-state"><div class="icon">⚠️</div>加载失败: ${e.message}</div>`;
        setStatus('预测加载失败 ✗');
    }
}

async function loadExperts() {
    const date = getDate();
    setStatus('正在加载专家...');
    document.getElementById('expert-date').textContent = date;
    try {
        const resp = await fetch(`/api/experts?date=${date}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        document.getElementById('expert-date').textContent =
            `${date}（专家: ${data.expert_count} | 反向: ${data.inverse_expert_count}）`;

        document.getElementById('expert-summary').innerHTML =
            `<span>🟢 正向专家: <strong>${data.expert_count}</strong> 人</span>` +
            `<span>🔴 反向专家: <strong>${data.inverse_expert_count}</strong> 人</span>` +
            `<span>📋 合计: <strong>${data.total}</strong> 条信号</span>`;

        if (!data.experts || data.experts.length === 0) {
            document.getElementById('experts-table').innerHTML =
                '<div class="empty-state"><div class="icon">🔍</div>当日无专家信号<br><small>可能原因：数据不足或没有满足阈值条件的用户</small></div>';
            return;
        }

        const typeLabel = t => t === 'expert' ? '🟢 专家' : '🔴 反向专家';
        const typeClass = t => t === 'expert' ? 'badge-expert' : 'badge-inverse';
        const dirClass = d => d === 'Bullish' ? 'badge-bullish' : 'badge-bearish';
        const dirLabel = d => d === 'Bullish' ? '📈 看涨' : '📉 看跌';

        const cols = [
            { key: 'user_id', label: '用户' },
            { key: 'stock', label: '股票', fmt: v => `<strong>${v}</strong>` },
            { key: 'expert_type', label: '类型', fmt: v => `<span class="badge ${typeClass(v)}">${typeLabel(v)}</span>` },
            { key: 'predicted_direction', label: '方向', fmt: v => `<span class="badge ${dirClass(v)}">${dirLabel(v)}</span>` },
            { key: 'accuracy_recent', label: '近期准确率', fmt: v => fmtPct1(v) },
            { key: 'accuracy_long', label: '长期准确率', fmt: v => fmtPct1(v) },
        ];
        document.getElementById('experts-table').innerHTML = buildTable(data.experts, cols, 30);
        setStatus('专家加载完成 ✓');
    } catch (e) {
        document.getElementById('experts-table').innerHTML =
            `<div class="empty-state"><div class="icon">⚠️</div>加载失败: ${e.message}</div>`;
        setStatus('专家加载失败 ✗');
    }
}

async function loadBacktest() {
    const endDate = getDate();
    const startDate = new Date(endDate);
    startDate.setDate(startDate.getDate() - 90);
    const start = startDate.toISOString().split('T')[0];

    try {
        const resp = await fetch(`/api/backtest?start=${start}&end=${endDate}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        const arClass = data.annualized_return > 0 ? 'green' : 'red';
        document.querySelector('#metrics-summary').innerHTML = `
            <div class="metric"><div class="value ${arClass}">${fmtPct1(data.annualized_return)}</div><div class="label">年化收益率</div></div>
            <div class="metric"><div class="value">${data.sharpe_ratio.toFixed(2)}</div><div class="label">夏普比率</div></div>
            <div class="metric"><div class="value">${fmtPct(data.mean_ic)}</div><div class="label">日均 IC</div></div>
            <div class="metric"><div class="value red">${fmtPct1(data.max_drawdown)}</div><div class="label">最大回撤</div></div>
        `;

        const cumReturns = data.cumulative_returns || [];
        if (cumReturns.length === 0) {
            document.getElementById('chart-container').innerHTML =
                '<div class="empty-state"><div class="icon">📉</div>暂无回测数据</div>';
            return;
        }

        if (returnsChart) returnsChart.destroy();
        const ctx = document.getElementById('returnsChart').getContext('2d');
        const labels = cumReturns.map((_, i) => i + 1);
        const finalVal = cumReturns[cumReturns.length - 1];
        const lineColor = finalVal >= 1 ? '#28a745' : '#dc3545';

        returnsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: '累计收益',
                    data: cumReturns,
                    borderColor: lineColor,
                    backgroundColor: finalVal >= 1 ? 'rgba(40,167,69,0.08)' : 'rgba(220,53,69,0.08)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                    borderWidth: 2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: ctx => `累计: ${(ctx.parsed.y * 100).toFixed(2)}%`,
                            title: ctx => `交易日 #${ctx[0].label}`,
                        }
                    }
                },
                scales: {
                    x: { display: true, title: { display: true, text: '交易日', color: '#888' }, ticks: { maxTicksLimit: 15, color: '#aaa' }, grid: { display: false } },
                    y: {
                        ticks: { callback: v => (v * 100).toFixed(0) + '%', color: '#aaa' },
                        grid: { color: '#f0f0f0' },
                    }
                }
            }
        });

        document.getElementById('chart-legend').textContent =
            `${data.n_trading_days} 个交易日 · 起止: ${start} ~ ${endDate} · ICIR: ${data.icir.toFixed(2)}`;
    } catch (e) {
        console.error('回测加载失败:', e);
        document.getElementById('chart-container').innerHTML =
            `<div class="empty-state"><div class="icon">⚠️</div>回测加载失败: ${e.message}</div>`;
    }
}

async function loadSystemStatus() {
    try {
        const stocksResp = await fetch('/api/stocks');
        const stocksData = await stocksResp.json();

        const date = getDate();
        const expertsResp = await fetch(`/api/experts?date=${date}`);
        const expertsData = await expertsResp.json();

        const predsResp = await fetch(`/api/predictions?date=${date}`);
        const predsData = await predsResp.json();
        const expertPreds = predsData.predictions.filter(p => p.signal_source === 'expert').length;

        document.getElementById('sys-stocks').textContent = stocksData.count;
        document.getElementById('sys-experts').textContent = expertsData.total;
        document.getElementById('sys-coverage').textContent = expertPreds + '/' + stocksData.count;
        document.getElementById('sys-posts').textContent = '—';
    } catch (e) {
        // System status is non-critical
    }
}

async function triggerCollect() {
    setStatus('正在采集数据...');
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
        setStatus('采集连接失败，请重试');
    };
}

// ── 表格构建工具 ──
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
        html += `<div style="text-align:center;padding:8px;color:#888;font-size:12px;">显示前 ${maxRows} 条，共 ${rows.length} 条</div>`;
    }
    return html;
}

// ── 自动刷新 (每5分钟) ──
setInterval(() => {
    loadPredictions();
    loadExperts();
    loadBacktest();
    loadSystemStatus();
}, 300000);
