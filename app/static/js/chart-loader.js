/* ============================================================
   chart-loader.js — resilient Chart.js loader
   Prefers the locally-vendored copy (survives blocked CDNs and
   offline use), then falls back to two independent CDNs.
   Consumers call whenChartReady(fn) / onChartFailed(fn).
   ============================================================ */
(function () {
    'use strict';

    var readyCallbacks = [];
    var failedCallbacks = [];
    var loaded = false;
    var failed = false;

    function fireReady() {
        if (loaded) return;
        loaded = true;
        readyCallbacks.forEach(function (cb) { try { cb(); } catch (e) { console.error(e); } });
        readyCallbacks = [];
    }

    function fireFailed() {
        if (failed || loaded) return;
        failed = true;
        failedCallbacks.forEach(function (cb) { try { cb(); } catch (e) { console.error(e); } });
        failedCallbacks = [];
    }

    window.whenChartReady = function (cb) {
        if (loaded || window.Chart) { loaded = true; cb(); }
        else readyCallbacks.push(cb);
    };

    window.onChartFailed = function (cb) {
        if (failed) cb();
        else failedCallbacks.push(cb);
    };

    // Cache-busting token is injected by the template as window.ASSET_V
    var v = window.ASSET_V ? ('?v=' + window.ASSET_V) : '';
    var sources = [
        '/static/vendor/chart.umd.js' + v,
        'https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js',
        'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js'
    ];

    (function loadNext(i) {
        if (i >= sources.length) {
            console.error('Chart.js failed to load from all sources (local + CDNs).');
            fireFailed();
            return;
        }
        var s = document.createElement('script');
        s.src = sources[i];
        s.onload = fireReady;
        s.onerror = function () { loadNext(i + 1); };
        document.head.appendChild(s);
    })(0);
})();
