/* ============================================================
   dashboard.js — all dashboard widgets.
   Sections:
     1. helpers + form state
     2. prediction + confidence range
     3. grid balance card
     4. 24-hour demand profile (peak / valley / ramp)
     5. what-if explorer
     6. feature importance
     7. train-vs-test metrics mini chart
     8. retrain + init
   ============================================================ */
(function () {
    'use strict';

    /* ---------- 1. Helpers + form state ---------- */
    var $ = function (id) { return document.getElementById(id); };

    function fmtMW(v, digits) {
        if (v === null || v === undefined || isNaN(v)) return '—';
        return Number(v).toLocaleString('en-US', {
            minimumFractionDigits: 0,
            maximumFractionDigits: digits === undefined ? 1 : digits
        });
    }

    function showError(el, err) {
        el.textContent = window.Chart || !err.chartRelated
            ? err.message
            : 'Charting library failed to load. Check your connection or ad blocker, then reload the page.';
        el.classList.add('show');
    }

    var form = $('predictionForm');
    var hourInput = $('hour');
    var hourNumberInput = $('hour_number');

    hourInput.addEventListener('input', function () {
        var h = parseInt(hourInput.value, 10);
        if (!isNaN(h)) hourNumberInput.value = h + 1;
    });

    function getFormValues() {
        var fd = new FormData(form);
        return {
            demand_forecast_mw: parseFloat(fd.get('demand_forecast_mw')),
            net_generation_mw: parseFloat(fd.get('net_generation_mw')),
            total_interchange_mw: parseFloat(fd.get('total_interchange_mw')),
            hour_number: parseInt(fd.get('hour_number'), 10),
            hour: parseInt(fd.get('hour'), 10),
            day_of_week: parseInt(fd.get('day_of_week'), 10),
            month: parseInt(fd.get('month'), 10),
            balancing_authority: fd.get('balancing_authority'),
            sub_region: fd.get('sub_region'),
            season: fd.get('season')
        };
    }

    /* ---------- 2. Prediction + confidence range ---------- */
    function setRange(lower, predicted, upper) {
        var margin = Math.max((upper - lower) * 0.4, 1);
        var domainMin = lower - margin;
        var domainMax = upper + margin;
        var pct = function (v) { return ((v - domainMin) / (domainMax - domainMin)) * 100; };

        $('rangeBand').style.left = pct(lower) + '%';
        $('rangeBand').style.width = (pct(upper) - pct(lower)) + '%';
        $('rangeMarker').style.left = pct(predicted) + '%';
        $('lowerLabel').textContent = fmtMW(lower) + ' MW';
        $('upperLabel').textContent = fmtMW(upper) + ' MW';
    }

    async function runPredict(data) {
        var loading = $('loading');
        var errorRow = $('predictError');
        var resultPanel = $('resultPanel');
        var predictBtn = $('predictBtn');

        loading.classList.add('show');
        errorRow.classList.remove('show');
        predictBtn.disabled = true;

        try {
            var result = await API.post('/predict/xgboost', data);
            $('predictedValue').textContent = fmtMW(result.predicted_demand_mw);
            setRange(result.lower_bound_mw, result.predicted_demand_mw, result.upper_bound_mw);
            resultPanel.classList.add('show');
            updateBalance(data.net_generation_mw, result.predicted_demand_mw);
        } catch (err) {
            resultPanel.classList.remove('show');
            showError(errorRow, err);
            resetBalance();
        } finally {
            loading.classList.remove('show');
            predictBtn.disabled = false;
        }
    }

    form.addEventListener('submit', function (e) {
        e.preventDefault();
        refreshAll();
    });

    /* ---------- 3. Grid balance card ---------- */
    function resetBalance() {
        $('balanceValue').textContent = '—';
        $('balancePill').textContent = '—';
        $('balancePill').className = 'pill';
        $('balanceNote').textContent = 'Run a prediction to see the margin.';
    }

    function updateBalance(netGen, predicted) {
        if (!isFinite(netGen) || !isFinite(predicted) || predicted <= 0) { resetBalance(); return; }

        var margin = netGen - predicted;
        var pct = (margin / predicted) * 100;
        var wrap = $('balanceWrap');
        var pill = $('balancePill');

        $('balanceValue').textContent = (margin >= 0 ? '+' : '−') + fmtMW(Math.abs(margin), 0);

        wrap.classList.remove('surplus', 'tight', 'deficit');
        if (pct >= 3) {
            wrap.classList.add('surplus');
            pill.textContent = 'Surplus';
            pill.className = 'pill pill-success';
        } else if (pct > -3) {
            wrap.classList.add('tight');
            pill.textContent = 'Tight';
            pill.className = 'pill pill-warn';
        } else {
            wrap.classList.add('deficit');
            pill.textContent = 'Deficit';
            pill.className = 'pill pill-danger';
        }

        var maxVal = Math.max(netGen, predicted);
        $('genBar').style.width = Math.max((netGen / maxVal) * 100, 2) + '%';
        $('demBar').style.width = Math.max((predicted / maxVal) * 100, 2) + '%';
        $('genNum').textContent = fmtMW(netGen, 0) + ' MW';
        $('demNum').textContent = fmtMW(predicted, 0) + ' MW';

        $('balanceNote').textContent =
            'Entered net generation covers ' + fmtMW((netGen / predicted) * 100, 1) + '% of predicted demand (' +
            (margin >= 0 ? 'a ' + fmtMW(Math.abs(pct), 1) + '% cushion' : 'a ' + fmtMW(Math.abs(pct), 1) + '% shortfall to import or dispatch') + ').';
    }

    /* ---------- 4. 24-hour demand profile ---------- */
    var dayCurveChart = null;

    async function loadDayCurve(baseValues) {
        var errorRow = $('dayCurveError');
        errorRow.classList.remove('show');

        try {
            var result = await API.post('/api/whatif', Object.assign({}, baseValues, { sweep_by: 'hour' }));
            renderDayCurve(result);
        } catch (err) {
            err.chartRelated = true;
            showError(errorRow, err);
        }
    }

    function renderDayCurve(result) {
        var points = result.points;
        var values = points.map(function (p) { return p.predicted_demand_mw; });
        var lower = points.map(function (p) { return p.lower_bound_mw; });
        var upper = points.map(function (p) { return p.upper_bound_mw; });
        var labels = points.map(function (p) { return p.label + ':00'; });

        // Peak / valley / ramp
        var peakIdx = 0, valleyIdx = 0;
        values.forEach(function (v, i) {
            if (v > values[peakIdx]) peakIdx = i;
            if (v < values[valleyIdx]) valleyIdx = i;
        });
        var peak = values[peakIdx], valley = values[valleyIdx];
        var ramp = peak - valley;

        $('peakChip').textContent = fmtMW(peak, 0) + ' MW at ' + labels[peakIdx];
        $('valleyChip').textContent = fmtMW(valley, 0) + ' MW at ' + labels[valleyIdx];
        $('rampChip').textContent = fmtMW(ramp, 0) + ' MW (' + (valley > 0 ? fmtMW((ramp / valley) * 100, 1) : '—') + '% swing)';

        // Contextualize MAE against the average predicted load of this profile
        var metricsEl = $('metricsData');
        var maeCtx = $('maeCtx');
        if (metricsEl && maeCtx) {
            var avg = values.reduce(function (a, b) { return a + b; }, 0) / values.length;
            var mae = parseFloat(metricsEl.dataset.mae);
            if (avg > 0 && isFinite(mae)) {
                maeCtx.textContent = '≈ ' + (mae / avg * 100).toFixed(1) + '% of avg predicted load in this profile';
            }
        }

        var pointRadius = values.map(function (_, i) { return (i === peakIdx || i === valleyIdx) ? 4 : 0; });
        var pointColors = values.map(function (_, i) {
            if (i === peakIdx) return '#c0362c';
            if (i === valleyIdx) return '#187a4b';
            return '#2f5fda';
        });

        if (dayCurveChart) dayCurveChart.destroy();
        dayCurveChart = new Chart($('dayCurveChart'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Upper bound', data: upper, borderColor: 'transparent',
                        backgroundColor: 'rgba(47, 95, 218, 0.08)', fill: '+1', pointRadius: 0, tension: 0.35
                    },
                    {
                        label: 'Lower bound', data: lower, borderColor: 'transparent',
                        backgroundColor: 'rgba(47, 95, 218, 0.08)', fill: false, pointRadius: 0, tension: 0.35
                    },
                    {
                        label: 'Predicted demand', data: values,
                        borderColor: '#2f5fda', borderWidth: 2, tension: 0.35, fill: false,
                        pointRadius: pointRadius, pointBackgroundColor: pointColors, pointBorderColor: pointColors
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                return ctx.dataset.label === 'Predicted demand'
                                    ? 'Predicted: ' + fmtMW(ctx.parsed.y) + ' MW' : null;
                            }
                        },
                        filter: function (ctx) { return ctx.dataset.label === 'Predicted demand'; }
                    }
                },
                scales: {
                    x: { grid: { display: false }, ticks: { font: { size: 10 }, maxTicksLimit: 12 } },
                    y: { grid: { color: '#eef0f4' }, ticks: { font: { size: 10 } } }
                }
            }
        });
    }

    /* ---------- 5. What-if explorer ---------- */
    var whatifChart = null;

    function getActiveSweep() {
        var active = document.querySelector('#sweepTabs .tab-btn.active');
        return active ? active.dataset.sweep : 'day_of_week';
    }

    async function loadWhatIf(sweepBy, baseValues) {
        var errorRow = $('whatifError');
        errorRow.classList.remove('show');

        try {
            var result = await API.post('/api/whatif', Object.assign({}, baseValues, { sweep_by: sweepBy }));
            renderWhatIf(result);
        } catch (err) {
            err.chartRelated = true;
            showError(errorRow, err);
        }
    }

    function renderWhatIf(result) {
        var labels = result.points.map(function (p) { return String(p.label); });
        var values = result.points.map(function (p) { return p.predicted_demand_mw; });
        var lower = result.points.map(function (p) { return p.lower_bound_mw; });
        var upper = result.points.map(function (p) { return p.upper_bound_mw; });
        var isCategorical = result.sweep_by === 'balancing_authority';

        if (whatifChart) whatifChart.destroy();
        whatifChart = new Chart($('whatifChart'), {
            type: isCategorical ? 'bar' : 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Upper bound', data: upper, borderColor: 'transparent',
                        backgroundColor: 'rgba(47, 95, 218, 0.10)', fill: '+1', pointRadius: 0, tension: 0.3
                    },
                    {
                        label: 'Lower bound', data: lower, borderColor: 'transparent',
                        backgroundColor: 'rgba(47, 95, 218, 0.10)', fill: false, pointRadius: 0, tension: 0.3
                    },
                    {
                        label: 'Predicted demand', data: values,
                        borderColor: '#2f5fda',
                        backgroundColor: isCategorical ? '#2f5fda' : 'rgba(47, 95, 218, 0.25)',
                        borderWidth: 2, pointRadius: isCategorical ? 0 : 2, tension: 0.3, fill: false,
                        borderRadius: isCategorical ? 4 : 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                return ctx.dataset.label === 'Predicted demand'
                                    ? 'Predicted: ' + fmtMW(ctx.parsed.y) + ' MW' : null;
                            }
                        },
                        filter: function (ctx) { return ctx.dataset.label === 'Predicted demand'; }
                    }
                },
                scales: {
                    x: { grid: { display: false }, ticks: { font: { size: 10 } } },
                    y: { grid: { color: '#eef0f4' }, ticks: { font: { size: 10 } } }
                }
            }
        });
    }

    $('sweepTabs').addEventListener('click', function (e) {
        var btn = e.target.closest('.tab-btn');
        if (!btn) return;
        document.querySelectorAll('#sweepTabs .tab-btn').forEach(function (b) { b.classList.remove('active'); });
        btn.classList.add('active');
        loadWhatIf(btn.dataset.sweep, getFormValues());
    });

    /* ---------- 6. Feature importance ---------- */
    async function loadFeatureImportance() {
        var errorRow = $('importanceError');
        try {
            var result = await API.get('/api/feature-importance');
            var top = result.features.slice(0, 8);

            new Chart($('importanceChart'), {
                type: 'bar',
                data: {
                    labels: top.map(function (f) { return f.feature; }),
                    datasets: [{
                        data: top.map(function (f) { return f.importance; }),
                        backgroundColor: '#2f5fda',
                        borderRadius: 4
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: '#eef0f4' }, ticks: { font: { size: 10 } } },
                        y: { grid: { display: false }, ticks: { font: { size: 11 } } }
                    }
                }
            });
        } catch (err) {
            err.chartRelated = true;
            showError(errorRow, err);
        }
    }

    /* ---------- 7. Train-vs-test metrics mini chart ---------- */
    function initMetricsChart() {
        var metricsEl = $('metricsData');
        if (!metricsEl) return;

        new Chart($('metricsChart'), {
            type: 'bar',
            data: {
                labels: ['MAE', 'RMSE'],
                datasets: [
                    { label: 'Train', data: [parseFloat(metricsEl.dataset.trainMae), parseFloat(metricsEl.dataset.trainRmse)], backgroundColor: '#c7d3f5' },
                    { label: 'Test', data: [parseFloat(metricsEl.dataset.mae), parseFloat(metricsEl.dataset.rmse)], backgroundColor: '#2f5fda' }
                ]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: { grid: { display: false }, ticks: { font: { size: 10 } } }
                }
            }
        });
    }

    /* ---------- 8. Retrain + init ---------- */
    $('retrainBtn').addEventListener('click', async function () {
        if (!confirm('Retrain the model with the latest data? This runs in the background and may take several minutes.')) return;
        var btn = $('retrainBtn');
        var loading = $('retrainLoading');
        btn.disabled = true;
        loading.classList.add('show');
        try {
            var result = await API.post('/train', {});
            alert(result.message);
        } catch (err) {
            alert('Error starting training: ' + err.message);
        } finally {
            btn.disabled = false;
            loading.classList.remove('show');
        }
    });

    function refreshAll() {
        var data = getFormValues();
        runPredict(data);
        loadDayCurve(data);
        loadWhatIf(getActiveSweep(), data);
    }

    whenChartReady(function () {
        initMetricsChart();
        loadFeatureImportance();
        refreshAll();
    });

    onChartFailed(function () {
        var msg = 'Charting library failed to load. Check your connection or ad blocker, then reload the page.';
        ['dayCurveError', 'whatifError', 'importanceError'].forEach(function (id) {
            var el = $(id);
            if (el) { el.textContent = msg; el.classList.add('show'); }
        });
    });
})();
