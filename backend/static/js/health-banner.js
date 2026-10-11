/**
 * Sensor health banner - fetches /api/dashboard-health-summary and renders
 * a device-grouped warning banner into the given container. Also shows the
 * comfort sensors (LK + Sonoff) from /api/comfort-sensors: silent, low battery,
 * weak signal or disagreeing sensors. Shared across all Theria pages that have
 * a #warning-container element.
 */
function comfortBannerHtml(data) {
    const problems = (data.sensors || []).filter(s => s.status !== 'ok');
    if (problems.length === 0) return '';
    const lines = problems.map(s => {
        const icon = s.status === 'fault' ? '🔴' : '🟡';
        const issues = s.issues.map(i => i.message).join('; ');
        const battery = s.battery_percent === null ? '' : ` (batteri ${Math.round(s.battery_percent)} %)`;
        return `<li>${icon} <strong>${s.name}</strong>${battery}: ${issues}</li>`;
    }).join('');
    const room = data.room_temperature_c === null
        ? 'Ingen frisk givare - planen väntar.'
        : `Planen använder rumstemperatur ${data.room_temperature_c.toFixed(1)} °C.`;
    return `
        <div class="warning-icon">🔋</div>
        <div class="warning-content">
            <strong>Komfortgivare</strong>
            <ul style="margin: 6px 0 0 0; padding-left: 18px;">${lines}</ul>
            <div style="margin-top: 6px;">${room}</div>
        </div>
    `;
}

async function loadComfortBanner(container) {
    try {
        const res = await fetch('api/comfort-sensors');
        if (!res.ok) return;
        const html = comfortBannerHtml(await res.json());
        let banner = container.querySelector('.comfort-banner');
        if (!html) {
            if (banner) banner.remove();
            return;
        }
        if (!banner) {
            banner = document.createElement('div');
            banner.className = 'warning-banner comfort-banner';
            container.appendChild(banner);
        }
        banner.innerHTML = html;
    } catch (error) {
        console.error('Error loading comfort sensor banner:', error);
    }
}

async function loadHealthBanner(containerId = 'warning-container') {
    const container = document.getElementById(containerId);
    if (!container) return;

    loadComfortBanner(container);

    try {
        const res = await fetch('api/dashboard-health-summary');
        const data = await res.json();

        if (!data.enabled || data.status === 'ok' || data.status === 'starting_up') {
            const existing = container.querySelector('.health-banner');
            if (existing) existing.remove();
            return;
        }

        const deviceLines = data.devices.map(device => {
            const icon = device.worst_status === 'unavailable' ? '🔴' : '🟡';
            const sensorList = device.sensors
                .map(s => `${s.entity_id} (${s.status}: ${s.message})`)
                .join(', ');
            return `<li>${icon} <strong>${device.device_name}</strong> - ${device.issue_count} sensor(s): ${sensorList}</li>`;
        }).join('');

        let banner = container.querySelector('.health-banner');
        if (!banner) {
            banner = document.createElement('div');
            banner.className = 'warning-banner health-banner';
            container.appendChild(banner);
        }
        banner.innerHTML = `
            <div class="warning-icon">⚠️</div>
            <div class="warning-content">
                <strong>Sensor Health Issues</strong>
                <ul style="margin: 6px 0 0 0; padding-left: 18px;">${deviceLines}</ul>
            </div>
        `;
    } catch (error) {
        console.error('Error loading health banner:', error);
    }
}
