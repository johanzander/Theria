
# Theria

## Overview

Theria is an open-source heating controller designed for cost-optimization and predictive climate management.

The service leverages day-ahead electricity prices to strategically plan and shift heating loads. By continuously learning each building's unique **thermal profile**—the specific relationship between energy input, heat retention, and external weather conditions—Theria proactively calculates the most economical heating schedule. This approach maintains consistent comfort while reducing energy bills.

## Key Features

### Day-Ahead Load Shifting

Leverages electricity prices published in advance to plan optimal heating cycles. By identifying the lowest-cost periods and utilizing your building's thermal mass as energy storage, the system maintains comfort while systematically avoiding peak-price periods.

### Multi-Zone Thermal Management

Manages independent temperature targets and comfort boundaries across multiple zones. Each room is treated as a distinct thermal environment, optimizing energy distribution based on localized demand and heat retention characteristics.

### Adaptive Thermal Modeling

Continuously learns your building's unique **thermal profile** by correlating heating output with temperature changes and external weather conditions. The model automatically refines heat retention rates and adjusts to seasonal shifts without manual calibration.

### Platform-Agnostic Integration

Designed as a Home Assistant add-on, Theria works with any climate entity or temperature sensor supported by the platform. Provides a unified control layer for diverse heating hardware, ensuring vendor-independent, future-proof operation.

### Operational Analytics

Provides direct visibility into energy consumption and cost performance. Detailed visualization tools track savings and system efficiency, enabling data-driven verification of optimization results.

## System Requirements

### Compatible Heating Systems

Theria works with any heating system integrated with Home Assistant that provides:

- **Climate Entities**: Thermostats or climate controls with target temperature setting
- **Temperature Sensors**: Current temperature measurements
- **Heating State** (recommended): `heating_power_request` or similar attribute indicating heating demand

### Supported Configurations

**Single Heating Source with Multiple Zones:**

- Central heat pump, boiler, or furnace
- Multiple zones sharing common heating capacity
- Each zone with independent climate control (thermostats, TRVs, zone valves)

**Examples:**

- Heat pump with radiators in multiple rooms
- Boiler with underfloor heating zones
- Mixed systems (radiators + floor heating)

### Required Integrations

1. **Home Assistant** - Core platform
2. **Nordpool Integration** - For electricity price data (or similar dynamic pricing source)
3. **Climate Entities** - One or more per zone
4. **Temperature Sensors** - At least one per zone

### Reference Implementation

See **[SYSTEM_SETUP.md](SYSTEM_SETUP.md)** for a complete example of Theria deployed on a multi-zone property with:

- Heat pump with multiple building zones
- Mixed heating types (radiators + floor heating)
- Different thermostat brands (Netatmo, Tado, LK)
- Nordpool electricity pricing (Sweden)

## How it works

### Important: Optimization Layer

**Theria is an optimization layer** that works on top of your existing heating control system. Your smart thermostats and heating system must already be functioning correctly with properly tuned heat curves. Theria enhances this working system by dynamically adjusting temperature targets based on electricity prices to reduce costs while maintaining comfort.

### Core Concept: Heat Capacitor Strategy

Theria implements a **heat capacitor** approach inspired by [PowerSaver](https://powersaver.no/nodes/ps-strategy-heat-capacitor.html), where each zone uses your building as a thermal battery:

1. **Pre-heat during cheap electricity hours** - Store energy in your building's thermal mass
2. **Coast during expensive hours** - Use stored thermal energy, avoid expensive heating
3. **Track savings per zone** - See exactly which zones save money and how much

### User-Defined Comfort Boundaries

For each zone, you define:

- **Target Comfort Temperature** ($\theta_{target}$): Your ideal temperature (e.g., 21°C)
- **Dynamic Deviation Band** ($\Delta\theta_{allowed}$): How much temperature can vary for savings (e.g., ±1°C)

**Example:**

- Target: 21°C
- Allowed range: 20-22°C
- During cheap hours: Heat to 22°C (store energy)
- During expensive hours: Let drop to 20°C (use stored energy)
- Result: Stay comfortable, reduce costs

### Key Inputs

**Market Data:**

- Published Nordpool day-ahead electricity prices (or similar dynamic pricing)
- Updated daily around 13:00 CET for next day

**Environmental Data:**

- Current outdoor temperature (from sensors)
- Current indoor temperature (from climate entities/sensors)

**System State:**

- Heating demand from climate entities (`heating_power_request`)
- Main system capacity (water temperature, setpoint)

### Control Strategy: Autonomous Zone Heat Capacitors

Each zone creates a **24-hour heating schedule** based on electricity prices using the heat capacitor strategy:

**Schedule Creation Process:**

1. **Analyze full 24h price data** - Receive Nordpool day-ahead prices (published daily ~13:00 CET)
2. **Identify buy-sell pairs** - Find cheap periods (heating opportunities) and expensive periods (coast opportunities)
3. **Calculate savings potential** - For each pair, compute if price differential justifies temperature adjustment cycle
4. **Consider thermal dynamics** - Account for time required to heat/cool zones by comfort deviation (e.g., ±1°C)
5. **Plan multiple cycles** - Schedule multiple pre-heat and coast periods throughout the day
6. **Execute schedule** - Every 15 minutes, adjust target temperature according to planned schedule

**Key Decision Factors:**

- Price differentials between cheap and expensive periods
- Zone thermal response time (learned adaptively)
- Maximum temperature deviation from comfort target
- Energy efficiency threshold (minimum savings to trigger cycle)
- Comfort constraints (temperature limits)

**Result:** Each zone autonomously manages its own heating schedule, pre-heating during cheap hours and coasting during expensive hours, all while staying within user-defined comfort boundaries.

### Thermal Model: Adaptive Learning

Instead of requiring extensive setup and calibration, each zone **learns automatically**:

**Every control cycle (15 minutes):**

1. **Predict**: "If I heat now, temperature will rise X°C in 15 minutes"
2. **Execute**: Set target temperature, heating activates
3. **Measure**: "Temperature actually rose Y°C"
4. **Learn**: Adjust model based on error (Y - X)
5. **Adapt**: Use improved model for next prediction

**Benefits:**

- Works from day 1 with rough initial guesses
- Continuously adapts to seasonal changes, occupancy, system aging
- No offline training or data collection required
- Handles unknown factors automatically

**What it learns:**

- How fast the zone heats up (depends on heating power, outdoor temp)
- How fast the zone cools down (depends on insulation, outdoor temp)
- Zone-specific characteristics (thermal mass, heat loss)

### Performance Tracking

**Per-Zone Analytics:**

- Energy consumed (kWh)
- Cost paid (currency)
- Baseline cost (what naive control would cost)
- **Savings** (baseline - actual)
- Savings percentage

**Compare zones:**
"Zone A saved 15% (€45), Zone B saved 8% (€20), Zone C saved 22% (€65)"

## Hierarchical Control Architecture

Theria manages a **cascaded multi-zone heating system** where multiple zones share a common heating source. The architecture consists of autonomous zone controllers and a reactive main system coordinator.

### Autonomous Zone Heat Capacitors

Each zone operates as an **independent heat capacitor** controller, inspired by the [PowerSaver Heat Capacitor strategy](https://powersaver.no/nodes/ps-strategy-heat-capacitor.html):

**Zone Controller Responsibilities:**

- **Reads Nordpool prices** autonomously (published hourly, day-ahead market)
- **Creates 24-hour heating schedule**: Analyzes full day price data to identify optimal pre-heat and coast periods
- **Decides when to pre-heat**: During cheap electricity periods before expensive hours
- **Decides when to coast**: During expensive periods, using stored thermal energy
- **Tracks own savings**: Independent financial analysis per zone
- **Learns adaptively**: Real-time adjustment of thermal response model

**Schedule Creation Logic:**

``` python
1. Receive 24h Nordpool price data (updated daily ~13:00)
2. Identify cheap-expensive price pairs (buy-sell opportunities)
3. Calculate savings potential: (expensive_price - cheap_price) × energy
4. Filter pairs where savings > efficiency_threshold
5. Plan pre-heat timing considering thermal response time
6. Generate minute-resolution temperature setpoint schedule
7. Execute: Every 15 min, apply scheduled target temperature
```

### Main Heating System Coordinator

The main heating system is **reactive**, providing capacity based on zone demands:

**Coordinator Responsibilities:**

- **Monitors**: Aggregate heating demand from all zones (via `heating_power_request`)
- **Controls**: Main system climate setpoint (e.g., `climate.room_temp_setpoint`)
- **Result**: Heat pump/boiler internal control adjusts output temperature accordingly (similar to Ngenic Tune approach)

**Coordination Logic:**

``` python
If majority_of_zones_demanding_heat:
    → Increase main system setpoint (ensure capacity available)

Elif no_zones_demanding_heat:
    → Decrease main system setpoint (reduce energy waste)
```

**Note:** The coordinator adjusts the main system's climate setpoint. The heat pump or boiler's internal control logic then responds by adjusting output temperature (e.g., radiator forward temperature) based on its own heat curve and control algorithms.

### Control Flow Example

**Scenario:** Electricity price rises from 0.50 SEK/kWh (06:00-07:59) to 1.50 SEK/kWh (08:00+)

**05:45 - Zone anticipates:**

- Zone controller reads prices: Cheap period coming, expensive period after
- Decision: "I will pre-heat during 06:00-07:59"

**06:00 - Pre-heat starts:**

- Zone raises target: 21°C → 23°C
- Climate entities request heat: `heating_power_request = 100%`
- Main system detects demand, raises main setpoint: 40°C → 45°C
- Heat pump/boiler responds by increasing output temperature
- Zone heats up: 21°C → 22.5°C (storing thermal energy)

**08:00 - Expensive period starts:**

- Zone reads new price: 1.50 SEK (expensive!)
- Decision: "I will coast"
- Zone lowers target: 23°C → 21°C
- Climate entities stop heating: `heating_power_request = 0%`
- Main system detects low demand, lowers main setpoint: 45°C → 35°C
- Heat pump/boiler reduces output temperature
- Zone coasts down: 22.5°C → 21°C (using stored energy)
- **Result:** Avoided expensive heating, savings tracked

### Key Benefits

✅ **Decentralized Intelligence** - Each zone makes autonomous decisions
✅ **Independent Analytics** - Track savings per zone (which zones perform best?)
✅ **No Central Coordination** - Zones don't conflict, they just demand capacity
✅ **Scalable** - Add zones without architectural changes
✅ **Transparent** - Clear cause-effect relationship between prices and actions

### Adaptive Learning

Instead of fitting thermal parameters once and using fixed values, each zone continuously learns:

**Every control cycle (15 minutes):**

1. **Predict** temperature change based on current model
2. **Execute** heating decision
3. **Measure** actual temperature change
4. **Learn** from prediction error (update model)
5. **Adapt** for next cycle

This real-time adaptation handles:

- Seasonal changes (insulation effectiveness)
- Occupancy patterns (internal heat gains)
- System aging (heat pump efficiency)
- Unknown factors (automatically compensates)

**Result:** The model always reflects current reality, without requiring extensive historical data collection.

## Implementation Notes

### Control Loop Timing

- **Zone Controllers**: 15-minute control cycles
  - Check electricity prices
  - Update target temperature based on heat capacitor logic
  - Measure and learn from actual temperature response

- **Main System Coordinator**: 15-minute cycles (can match zone timing)
  - Monitor aggregate zone demands
  - Adjust main setpoint to provide capacity

### 24-Hour Heating Schedule

- **Full day schedule creation** using Nordpool day-ahead prices
  - Receives complete 24-hour price data (published daily ~13:00 CET)
  - Creates comprehensive heating plan with multiple pre-heat/coast cycles
  - Updates schedule daily when new price data becomes available
  - Executes schedule in 15-minute control intervals throughout the day

**Note:** While zones can only heat/cool within realistic timeframes (typically 2-6 hours), the 24-hour schedule ensures optimal timing across the entire day, identifying all beneficial buy-sell price pairs.

### Adaptive Learning Parameters

**Initial Values:**

- Start with rough estimates (heating/cooling rates)
- System learns actual values within first few days
- Bad initial guesses corrected automatically

**Learning Rate:**

- Slow adaptation (0.05-0.1) prevents instability
- Recent data weighted higher than old data
- Outlier rejection for sensor errors

### Energy Monitoring

**Recommended for analytics:**

- Main system energy consumption (heat pump, boiler)
- Per-zone consumption (if available)
- Enables accurate savings calculation and validation

**Fallback:**

- Can estimate savings from price differences alone
- Less accurate but still provides useful insights

### Integration Requirements

**Minimum:**

- Climate entities with `set_temperature` service
- Temperature sensors (current readings)
- Nordpool integration (or similar pricing)

**Recommended:**

- Climate entities with `heating_power_request` attribute
- Main system control (setpoint adjustment)
- Energy monitoring sensors

### Technical Details

For architecture diagrams, sequence flows, and implementation phases, see:

- **[architecture.md](architecture.md)** - System design and control flow
- **[implementation-plan.md](implementation-plan.md)** - Development roadmap
