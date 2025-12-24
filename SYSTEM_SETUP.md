# Example System Setup

**Property:** Johan's House (Sweden)
**Location:** Multi-building property with shared heating system
**Last Updated:** 2025-12-23

---

## Property Layout

### Buildings

- **Main House**: Two-floor stone building (300 sqm) + basement
- **Boutique**: 35 sqm commercial space, junction with main house
- **Guest House**: 60 sqm separate building

### Heating Distribution

All buildings share a common heating source (IVT AirX heat pump):

- **Main House - Ground Floor**: Floor heating with LK ARC Hub control
- **Main House - Upper Floor**: Floor heating with LK IC2 control (no remote)
- **Boutique**: Water radiators with Netatmo smart thermostats
- **Guest House**: Water radiators with Tado smart thermostats

---

## Main Heating System

### Heat Pump

- **Model**: IVT AirX 170 + Airbox E9 130-170 (Rego 2000)
- **Type**: Air-to-water heat pump
- **Capacity**: Provides heating and domestic hot water
- **Water Tank**: External 250L tank
- **Control**: Husdata H60 device (Home Assistant integration)

### Key Sensors (H60 Main System)

**Temperature & Control:**

- `sensor.outdoor` - Outdoor temperature (external sensor)
- `climate.room_temp_setpoint` - Master setpoint control
- `sensor.radiator_forward` - Forward water temperature (all zones)
- `sensor.heating_setpoint` - Current heating setpoint
- `sensor.heat_carrier_forward` - Heat carrier forward temp
- `sensor.heat_carrier_return` - Heat carrier return temp
- `sensor.compressor_speed_2` - Compressor speed (0-100%)
- `binary_sensor.switch_valve_1` - Hot water (True) vs Heating (False)

**Energy Monitoring - Compressor:**

- `sensor.h60_compr_cons_heating` - Heating consumption
- `sensor.h60_compr_cons_hotwat` - Hot water consumption
- `sensor.h60_compr_consump_tot` - Total consumption

**Energy Monitoring - Auxiliary:**

- `sensor.h60_aux_cons_hot_water`
- `sensor.h60_aux_cons_heating`
- `sensor.h60_aux_consumption_tot`

**Energy Monitoring - Supplementary:**

- `sensor.h60_supp_energy_heating`
- `sensor.h60_supp_energy_hotwater`
- `sensor.h60_supp_energy_tot`

---

## Zone Configurations

### Zone 1: Boutique (35 sqm)

**Heating Type:** 2× water radiators (Netatmo)
**Radiator Specs:** 2 × 1060W at ΔT=30°C

**Climate Entities:**

- `climate.vantsidan`
  - Attributes: `current_temperature`, `heating_power_request`, `target_temperature`
- `climate.klippsidan`
  - Attributes: `current_temperature`, `heating_power_request`, `target_temperature`

**Supplementary Temperature Sensors:**

- `sensor.schamponeringstolen_temperature` (floor sensor)
- `sensor.butik_temperature_2` (ceiling sensor)

**Schedule:**

```yaml
- time: "04:00"
  target: 20.0
  deviation: 1.0

- time: "06:00"
  target: 22.0
  deviation: 1.0

- time: "19:00"
  target: 18.0
  deviation: 2.0
```

**Theria Configuration:**

```yaml
zones:
  - id: "boutique"
    name: "Boutique"
    climate_entities:
      - "climate.vantsidan"
      - "climate.klippsidan"
    temp_sensors:
      - "climate.vantsidan"  # Uses current_temperature
      - "climate.klippsidan"
      - "sensor.schamponeringstolen_temperature"
      - "sensor.butik_temperature_2"
    radiator_spec:
      count: 2
      nominal_power: 1060.0  # W per radiator
      nominal_delta_t: 30.0  # °C
      exponent: 1.3
    enabled: true
```

---

### Zone 2: First Floor (Main House)

**Heating Type:** Floor heating (LK control)

**Library:**

- `climate.lk_bibliotek_thermostat`
- `sensor.lk_bibliotek_temperature`

**Living Room:**

- `climate.lk_vardagsrum_thermostat`
- `sensor.lk_vardagsrum_temperature`
- `sensor.sonoff_a480045bf5_temperature` (supplementary)

**Guest Room:**

- `sensor.gastrum_temperature_2`

**Theria Configuration:**

```yaml
zones:
  - id: "first_floor"
    name: "First Floor"
    climate_entities:
      - "climate.lk_bibliotek_thermostat"
      - "climate.lk_vardagsrum_thermostat"
    temp_sensors:
      - "sensor.lk_bibliotek_temperature"
      - "sensor.lk_vardagsrum_temperature"
      - "sensor.sonoff_a480045bf5_temperature"
      - "sensor.gastrum_temperature_2"
    enabled: true
```

---

### Zone 3: Guest House (60 sqm)

**Heating Type:** Water radiators (Tado)

**Climate Entities:**

- `climate.bedroom`
- `climate.hallway`
- `climate.kitchen`
- `climate.tv_room_gh`
- `climate.storage_room`

**Theria Configuration:**

```yaml
zones:
  - id: "guest_house"
    name: "Guest House"
    climate_entities:
      - "climate.bedroom"
      - "climate.hallway"
      - "climate.kitchen"
      - "climate.tv_room_gh"
      - "climate.storage_room"
    enabled: true
```

---

## Electricity Pricing

**Provider:** Nordpool (Sweden SE4)
**Sensor:** `sensor.nordpool_kwh_se4_sek_2_10_025`
**Currency:** SEK
**Market:** Day-ahead spot prices (published daily ~13:00 for next day)

**Typical Price Range:**

- Cheap hours: 0.30 - 0.60 SEK/kWh
- Normal hours: 0.60 - 1.20 SEK/kWh
- Expensive hours: 1.20 - 2.50 SEK/kWh (winter peaks)

---

## System Identification Results

**Boutique Zone (as of Dec 2025):**

**Measured Parameters:**

- Thermal Mass (C): 2.33 MJ/K
- Heat Loss Coefficient (U): 51.7 W/K
- Time Constant (τ): ~12.5 hours

**Measured Transition Rates:**

- Heating rate: ~0.25-0.30°C/hour (both radiators active)
- Cooling rate: ~0.65-0.70°C/hour (heating off)

**Heating Activity (24h average):**

- vantsidan: 57.2% heating time
- klippsidan: 58.8% heating time
- Average power: 663W (range: 0-1830W)

---

## Notes

This is a **reference implementation** showing Theria deployed on a multi-zone property with:

- Shared heating source (heat pump)
- Mixed heating types (radiators + floor heating)
- Different control systems (Netatmo, Tado, LK)
- Nordpool electricity pricing

Your implementation may differ based on:

- Heating system type
- Number and size of zones
- Climate entity types
- Electricity market

See main README.MD for generic Theria documentation.
