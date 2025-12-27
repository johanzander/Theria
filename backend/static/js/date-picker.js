/**
 * Reusable date picker component for all Theria pages
 * Provides consistent date range selection across dashboard, zone, and system views
 */
export class TheriaDatePicker {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.onRangeChange = options.onRangeChange || (() => {});
        this.selectedTimeRange = 'today';
        this.customDateRange = null;
        this.datePickerInstance = null;

        // Load saved time range from localStorage
        this.loadSavedTimeRange();

        this.init();
    }

    init() {
        this.render();
        this.setupPresetButtons();
        this.setupCustomPicker();
        this.restoreSavedSelection();
    }

    /**
     * Load saved time range from localStorage
     */
    loadSavedTimeRange() {
        try {
            const saved = localStorage.getItem('theria_time_range');
            if (saved) {
                const { selectedTimeRange, customDateRange } = JSON.parse(saved);
                this.selectedTimeRange = selectedTimeRange || 'today';

                // Restore custom date range if it exists
                if (customDateRange && customDateRange.start && customDateRange.end) {
                    this.customDateRange = {
                        start: new Date(customDateRange.start),
                        end: new Date(customDateRange.end)
                    };
                }
            }
        } catch (e) {
            console.warn('Failed to load saved time range:', e);
            this.selectedTimeRange = 'today';
            this.customDateRange = null;
        }
    }

    /**
     * Save current time range to localStorage
     */
    saveTimeRange() {
        try {
            const data = {
                selectedTimeRange: this.selectedTimeRange,
                customDateRange: this.customDateRange ? {
                    start: this.customDateRange.start.toISOString(),
                    end: this.customDateRange.end.toISOString()
                } : null
            };
            localStorage.setItem('theria_time_range', JSON.stringify(data));
        } catch (e) {
            console.warn('Failed to save time range:', e);
        }
    }

    /**
     * Restore the saved selection in the UI after rendering
     */
    restoreSavedSelection() {
        if (this.customDateRange) {
            // Restore custom date range
            const input = document.getElementById('customDateRange');
            if (input && this.datePickerInstance) {
                this.datePickerInstance.setDate([this.customDateRange.start, this.customDateRange.end], false);

                // Format display text
                const options = { month: 'short', day: 'numeric' };
                const startStr = this.customDateRange.start.toLocaleDateString('en-US', options);
                const endStr = this.customDateRange.end.toLocaleDateString('en-US', options);
                const year = this.customDateRange.end.getFullYear();
                input.value = `${startStr} - ${endStr}, ${year}`;

                // Highlight custom range input
                document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
                input.classList.add('active');
            }
        } else {
            // Restore preset button selection
            document.querySelectorAll('.range-btn').forEach(btn => {
                if (btn.dataset.range === this.selectedTimeRange) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
        }
    }

    render() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error(`Container with id "${this.containerId}" not found`);
            return;
        }

        container.innerHTML = `
            <div class="time-range-selector">
                <button class="range-btn active" data-range="today">Today</button>
                <button class="range-btn" data-range="yesterday">Yesterday</button>
                <button class="range-btn" data-range="week">Week</button>
                <button class="range-btn" data-range="month">Month</button>
                <input type="text" id="customDateRange" class="custom-range-input"
                       placeholder="Custom range..." readonly style="width: 200px; margin-left: 10px;">
            </div>
        `;
    }

    setupPresetButtons() {
        document.querySelectorAll('.range-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.selectedTimeRange = btn.dataset.range;
                this.customDateRange = null;
                document.getElementById('customDateRange').classList.remove('active');
                document.getElementById('customDateRange').value = '';

                // Save to localStorage
                this.saveTimeRange();

                this.onRangeChange();
            });
        });
    }

    setupCustomPicker() {
        const input = document.getElementById('customDateRange');

        this.datePickerInstance = flatpickr(input, {
            mode: 'range',
            dateFormat: 'Y-m-d',
            maxDate: 'today',
            defaultDate: null,
            theme: 'dark',
            onChange: (selectedDates, dateStr, instance) => {
                if (selectedDates.length === 2) {
                    this.customDateRange = {
                        start: selectedDates[0],
                        end: selectedDates[1]
                    };

                    document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
                    input.classList.add('active');

                    const options = { month: 'short', day: 'numeric' };
                    const startStr = selectedDates[0].toLocaleDateString('en-US', options);
                    const endStr = selectedDates[1].toLocaleDateString('en-US', options);
                    const year = selectedDates[1].getFullYear();
                    input.value = `${startStr} - ${endStr}, ${year}`;

                    // Save to localStorage
                    this.saveTimeRange();

                    this.onRangeChange();
                }
            },
            onOpen: () => {
                if (!this.customDateRange) {
                    const end = new Date();
                    const start = new Date();
                    start.setDate(start.getDate() - 7);
                    this.datePickerInstance.setDate([start, end], false);
                }
            }
        });
    }

    buildChartDataUrl(baseUrl) {
        const now = new Date();
        let fromMs, toMs;

        if (this.customDateRange && this.customDateRange.start && this.customDateRange.end) {
            const from = new Date(this.customDateRange.start);
            from.setHours(0, 0, 0, 0);
            const to = new Date(this.customDateRange.end);
            to.setHours(23, 59, 59, 999);
            fromMs = from.getTime();
            toMs = to.getTime();
        } else if (this.selectedTimeRange === 'today') {
            const todayStart = new Date(now);
            todayStart.setHours(0, 0, 0, 0);
            fromMs = todayStart.getTime();
            toMs = now.getTime();
        } else if (this.selectedTimeRange === 'yesterday') {
            const yesterdayStart = new Date(now);
            yesterdayStart.setDate(yesterdayStart.getDate() - 1);
            yesterdayStart.setHours(0, 0, 0, 0);
            const yesterdayEnd = new Date(yesterdayStart);
            yesterdayEnd.setHours(23, 59, 59, 999);
            fromMs = yesterdayStart.getTime();
            toMs = yesterdayEnd.getTime();
        } else if (this.selectedTimeRange === 'week') {
            const dayOfWeek = now.getDay();
            const daysToMonday = (dayOfWeek === 0 ? 6 : dayOfWeek - 1);
            const monday = new Date(now);
            monday.setDate(monday.getDate() - daysToMonday);
            monday.setHours(0, 0, 0, 0);
            fromMs = monday.getTime();
            toMs = now.getTime();
        } else if (this.selectedTimeRange === 'month') {
            const monthStart = new Date(now.getFullYear(), now.getMonth(), 1);
            monthStart.setHours(0, 0, 0, 0);
            fromMs = monthStart.getTime();
            toMs = now.getTime();
        } else {
            const hours = this.getHoursFromRange();
            fromMs = now.getTime() - (hours * 60 * 60 * 1000);
            toMs = now.getTime();
        }

        return `${baseUrl}?from=${fromMs}&to=${toMs}`;
    }

    getHoursFromRange() {
        const rangeMap = {
            'today': 48,
            'yesterday': 48,
            'week': 168,
            'month': 744
        };

        if (this.customDateRange && this.customDateRange.start && this.customDateRange.end) {
            const diffMs = this.customDateRange.end - this.customDateRange.start;
            const diffHours = Math.ceil(diffMs / (1000 * 60 * 60));
            return diffHours + 24;
        }

        return rangeMap[this.selectedTimeRange] || 48;
    }

    getTimeRange() {
        return {
            selectedTimeRange: this.selectedTimeRange,
            customDateRange: this.customDateRange,
            hours: this.getHoursFromRange()
        };
    }

    getTimeBounds() {
        const now = new Date();
        let xAxisMin, xAxisMax;

        if (this.customDateRange && this.customDateRange.start && this.customDateRange.end) {
            const from = new Date(this.customDateRange.start);
            from.setHours(0, 0, 0, 0);
            const to = new Date(this.customDateRange.end);
            to.setHours(23, 59, 59, 999);
            xAxisMin = from.toISOString();
            xAxisMax = to.toISOString();
        } else if (this.selectedTimeRange === 'today') {
            const todayStart = new Date(now);
            todayStart.setHours(0, 0, 0, 0);
            xAxisMin = todayStart.toISOString();
            xAxisMax = now.toISOString();
        } else if (this.selectedTimeRange === 'yesterday') {
            const yesterdayStart = new Date(now);
            yesterdayStart.setDate(yesterdayStart.getDate() - 1);
            yesterdayStart.setHours(0, 0, 0, 0);
            const yesterdayEnd = new Date(yesterdayStart);
            yesterdayEnd.setHours(23, 59, 59, 999);
            xAxisMin = yesterdayStart.toISOString();
            xAxisMax = yesterdayEnd.toISOString();
        } else if (this.selectedTimeRange === 'week') {
            const dayOfWeek = now.getDay();
            const daysToMonday = (dayOfWeek === 0 ? 6 : dayOfWeek - 1);
            const monday = new Date(now);
            monday.setDate(monday.getDate() - daysToMonday);
            monday.setHours(0, 0, 0, 0);
            xAxisMin = monday.toISOString();
            xAxisMax = now.toISOString();
        } else if (this.selectedTimeRange === 'month') {
            const monthStart = new Date(now.getFullYear(), now.getMonth(), 1);
            monthStart.setHours(0, 0, 0, 0);
            xAxisMin = monthStart.toISOString();
            xAxisMax = now.toISOString();
        } else {
            const hours = this.getHoursFromRange();
            const fromTime = new Date(now.getTime() - (hours * 60 * 60 * 1000));
            xAxisMin = fromTime.toISOString();
            xAxisMax = now.toISOString();
        }

        return { xAxisMin, xAxisMax };
    }
}
