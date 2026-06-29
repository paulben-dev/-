let returnsChart = null;

// Set default date to yesterday
document.addEventListener('DOMContentLoaded', () => {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    document.getElementById('date-picker').value = yesterday.toISOString().split('T')[0];
    loadPredictions();
    loadExperts();
    loadBacktest();
});

function getDate() {
    return document.getElementById('date-picker').value;
}

function setStatus(msg) {
    document.getElementById('status').textContent = msg;
}

async function loadPredictions() {
    const date = getDate();
    setStatus('Loading predictions...');
    document.getElementById('pred-date').textContent = date;
    try {
        const resp = await fetch(`/api/predictions?date=${date}`);
        const data = await resp.json();
        document.getElementById('pred-date').textContent =
            `${date} (${data.expert_coverage} signals)`;

        if (!data.predictions || data.predictions.length === 0) {
            document.getElementById('predictions-table').innerHTML =
                '<p style="color:#888; text-align:center; padding:20px;">No predictions available</p>';
            return;
        }

        let html = '<table><thead><tr><th>Stock</th><th>Predicted Return</th><th>Signal Source</th></tr></thead><tbody>';
        data.predictions.slice(0, 20).forEach(p => {
            const retClass = p.predicted_return > 0 ? 'green' : 'red';
            const sourceClass = p.signal_source === 'expert' ? 'badge-expert-signal' : 'badge-momentum';
            html += `<tr>
                <td><strong>${p.stock}</strong></td>
                <td class="${retClass}">${(p.predicted_return * 100).toFixed(2)}%</td>
                <td><span class="badge ${sourceClass}">${p.signal_source}</span></td>
            </tr>`;
        });
        html += '</tbody></table>';
        document.getElementById('predictions-table').innerHTML = html;
        setStatus('Predictions loaded');
    } catch (e) {
        document.getElementById('predictions-table').innerHTML =
            `<p style="color:red;">Error: ${e.message}</p>`;
        setStatus('Error loading predictions');
    }
}

async function loadExperts() {
    const date = getDate();
    setStatus('Loading experts...');
    document.getElementById('expert-date').textContent = date;
    try {
        const resp = await fetch(`/api/experts?date=${date}`);
        const data = await resp.json();
        document.getElementById('expert-date').textContent =
            `${date} (${data.expert_count} experts, ${data.inverse_expert_count} inverse)`;

        if (!data.experts || data.experts.length === 0) {
            document.getElementById('experts-table').innerHTML =
                '<p style="color:#888; text-align:center; padding:20px;">No expert signals today</p>';
            return;
        }

        let html = '<table><thead><tr><th>User</th><th>Stock</th><th>Type</th><th>Direction</th><th>Recent Acc</th><th>Long Acc</th></tr></thead><tbody>';
        data.experts.forEach(e => {
            const typeClass = e.expert_type === 'expert' ? 'badge-expert' : 'badge-inverse';
            const dirClass = e.predicted_direction === 'Bullish' ? 'badge-bullish' : 'badge-bearish';
            html += `<tr>
                <td>${e.user_id}</td>
                <td><strong>${e.stock}</strong></td>
                <td><span class="badge ${typeClass}">${e.expert_type}</span></td>
                <td><span class="badge ${dirClass}">${e.predicted_direction}</span></td>
                <td>${(e.accuracy_recent * 100).toFixed(1)}%</td>
                <td>${(e.accuracy_long * 100).toFixed(1)}%</td>
            </tr>`;
        });
        html += '</tbody></table>';
        document.getElementById('experts-table').innerHTML = html;
        setStatus('Experts loaded');
    } catch (e) {
        document.getElementById('experts-table').innerHTML =
            `<p style="color:red;">Error: ${e.message}</p>`;
        setStatus('Error loading experts');
    }
}

async function loadBacktest() {
    const endDate = getDate();
    const startDate = new Date(endDate);
    startDate.setDate(startDate.getDate() - 90);
    const start = startDate.toISOString().split('T')[0];

    try {
        const resp = await fetch(`/api/backtest?start=${start}&end=${endDate}`);
        const data = await resp.json();

        document.querySelector('#metrics-summary').innerHTML = `
            <div class="metric"><div class="value ${data.annualized_return > 0 ? 'green' : 'red'}">${(data.annualized_return * 100).toFixed(1)}%</div><div class="label">Annualized Return</div></div>
            <div class="metric"><div class="value">${data.sharpe_ratio.toFixed(2)}</div><div class="label">Sharpe Ratio</div></div>
            <div class="metric"><div class="value">${(data.mean_ic * 100).toFixed(2)}%</div><div class="label">Mean IC</div></div>
            <div class="metric"><div class="value red">${(data.max_drawdown * 100).toFixed(1)}%</div><div class="label">Max Drawdown</div></div>
        `;

        if (returnsChart) returnsChart.destroy();
        const ctx = document.getElementById('returnsChart').getContext('2d');
        const cumReturns = data.cumulative_returns || [];
        const labels = cumReturns.map((_, i) => i);

        returnsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Cumulative Return',
                    data: cumReturns,
                    borderColor: cumReturns.length > 0 && cumReturns[cumReturns.length - 1] >= 1 ? '#28a745' : '#dc3545',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    fill: true,
                    tension: 0.3,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: {
                        ticks: { callback: v => (v * 100).toFixed(1) + '%' }
                    }
                }
            }
        });
        setStatus('Backtest loaded');
    } catch (e) {
        console.error('Backtest error:', e);
    }
}

async function triggerCollect() {
    setStatus('Collecting data...');
    const endDate = getDate();
    const startDate = new Date(endDate);
    startDate.setDate(startDate.getDate() - 7);
    try {
        const resp = await fetch(`/api/collect?start=${startDate.toISOString().split('T')[0]}&end=${endDate}`, { method: 'POST' });
        const data = await resp.json();
        setStatus(`Collected: ${data.results.prices} prices, ${data.results.stocktwits} StockTwits, ${data.results.reddit} Reddit posts`);
        loadPredictions();
        loadExperts();
    } catch (e) {
        setStatus('Collection failed: ' + e.message);
    }
}

// Auto-refresh every 5 minutes
setInterval(() => {
    loadPredictions();
    loadExperts();
}, 300000);
