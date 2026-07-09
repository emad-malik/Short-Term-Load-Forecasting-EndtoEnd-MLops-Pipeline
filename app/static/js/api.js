/* ============================================================
   api.js — thin fetch helpers shared by dashboard widgets.
   Every helper throws Error(detail) on non-2xx so callers can
   surface the FastAPI `detail` message directly.
   ============================================================ */
(function () {
    'use strict';

    async function handle(res) {
        if (!res.ok) {
            var body = await res.json().catch(function () { return {}; });
            throw new Error(body.detail || ('Request failed (' + res.status + ')'));
        }
        return res.json();
    }

    window.API = {
        get: function (url) {
            return fetch(url).then(handle);
        },
        post: function (url, payload) {
            return fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            }).then(handle);
        }
    };
})();
