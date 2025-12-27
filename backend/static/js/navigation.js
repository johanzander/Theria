/**
 * Dynamic navigation component for Theria
 * Generates navigation menu based on configured zones from the API
 */

/**
 * Load and render dynamic navigation
 * @param {string} currentPage - Current page type: 'dashboard', 'zone', or 'system'
 * @param {string|null} currentZoneId - Current zone ID if on a zone page
 */
export async function loadNavigation(currentPage = 'dashboard', currentZoneId = null) {
    try {
        // Fetch zones from API
        const res = await fetch('/api/zones');
        const data = await res.json();

        if (!data.zones || data.zones.length === 0) {
            console.warn('No zones configured');
            return;
        }

        // Build navigation HTML
        let navHTML = `
            <a href="/" ${currentPage === 'dashboard' ? 'class="active"' : ''} style="color: ${currentPage === 'dashboard' ? '#4fd1c5' : '#4fd1c5'};">
                ← Dashboard
            </a>
            <span style="color: #4a5568;">|</span>
            <a href="/system-insights" ${currentPage === 'system' ? 'class="active"' : ''} style="color: ${currentPage === 'system' ? '#4fd1c5' : '#4fd1c5'};">
                System Insights
            </a>
        `;

        // Add zone links dynamically
        const enabledZones = data.zones.filter(zone => zone.enabled !== false);

        if (enabledZones.length > 0) {
            navHTML += '<span style="color: #4a5568;">|</span>';
            navHTML += ' Zones: ';

            navHTML += enabledZones.map(zone => {
                const isActive = currentPage === 'zone' && zone.id === currentZoneId;
                const icon = zone.icon || '📊';
                return `
                    <a href="/zone-insights?zone_id=${zone.id}"
                       ${isActive ? 'class="active"' : ''}
                       style="color: ${isActive ? '#fff' : '#4fd1c5'}; margin: 0 5px;">
                        ${icon} ${zone.name}
                    </a>
                `;
            }).join(' ');
        }

        // Inject navigation into the page
        const container = document.querySelector('.breadcrumb');
        if (container) {
            container.innerHTML = navHTML;
        } else {
            console.warn('Navigation container (.breadcrumb) not found');
        }

    } catch (error) {
        console.error('Error loading navigation:', error);
    }
}

/**
 * Initialize navigation on page load
 * Automatically detects current page and zone from URL
 */
export function initNavigation() {
    // Detect current page type
    const path = window.location.pathname;
    const urlParams = new URLSearchParams(window.location.search);

    let currentPage = 'dashboard';
    let currentZoneId = null;

    if (path.includes('zone-insights')) {
        currentPage = 'zone';
        currentZoneId = urlParams.get('zone_id');
    } else if (path.includes('system-insights')) {
        currentPage = 'system';
    } else if (path === '/' || path.includes('index')) {
        currentPage = 'dashboard';
    }

    // Load navigation with detected context
    loadNavigation(currentPage, currentZoneId);
}
