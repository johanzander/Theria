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

**Confirmed against `home-analytics/backend/sensors.yaml` and the official Husdata H66 register
map (online.husdata.se/h-docs/C10.pdf, updated 2026-08-01). Entity IDs have no `h60_` prefix and
one has a typo baked into the real integration (`heat_carrier_forwrd`). Register IDs below are
the Husdata Modbus register (IDX), not the HA entity ID.**

**Temperature & Control:**

- `sensor.outdoor` (reg `0007`) - Outdoor temperature (external sensor)
- `sensor.radiator_forward` (reg `0002`) - Water feed out to the radiators/zones (distinct from
  heat carrier forward - this is downstream of the mixing, closer to what the zones actually get)
- `sensor.heat_carrier_forwrd` (reg `0004`) - HP's internal heat supply forward (note: "forwrd",
  not "forward" - typo baked into the real integration)
- `sensor.heat_carrier_return` (reg `0003`) - HP's internal heat carrier return
- `sensor.heating_setpoint` (reg `0107`) - Target temp for heating (read/write)
- `sensor.compressor_speed_2` - Compressor speed (% of max; the name is misleading, see below)
- `sensor.compressor_speed` - Power in kW, not electrical input (2026-11-08: the name and the unit do not match, and
  it integrates to more than the heat supplied; see `documentation/ivt-heat-pump.md`)
- `binary_sensor.switch_valve_1` (reg `1A07`) - `0`/off = Radiator heating, `1`/on = Hot water heating

**Energy Monitoring - Compressor (cumulative kWh counters):**

- `sensor.compr_cons_heating` (reg `5C55`) - Heating consumption
- `sensor.compr_cons_hotwat` (reg `5C56`) - Hot water consumption
- `sensor.compr_consump_tot` (reg `5C54`) - Total consumption

**Energy Monitoring - Auxiliary (cumulative kWh counters):**

- `sensor.aux_consumption_tot` (reg `5C57`) - Electrical additional heater, total
- Registers `5C58`/`5C59` (heating/hot-water split) exist on the pump but aren't in
  `home-analytics/backend/sensors.yaml` or Theria's config yet - add if the split becomes useful.

**Energy Monitoring - Delivered/Supplementary Energy (cumulative kWh counters):**

- `sensor.supp_energy_heating` (reg `5C52`) - Energy supplied for heating
- `sensor.supp_energy_hotwater` (reg `5C53`) - Energy supplied for hot water production
- `sensor.supp_energy_tot` (reg `5C51`) - Energy supplied, total

**Domestic Hot Water:**

- `sensor.warm_water_setpoint` (reg `0111`) - Target temp for warm tap water
- `sensor.warm_water_1_top` (reg `0009`) - Warm water tank, top sensor
- `sensor.warm_water_2_mid` (reg `000A`) - Warm water tank, mid sensor

**Also available on the pump but not yet tracked** (reg `6C60` Compr. Runtime hours,
`2C61` Compr. Starts counter, `6C50` Total op. time hours, `3104` Add heat status % usage) -
useful for reliability/health monitoring later, not needed for the current visibility pass.

Tracked in Theria via `core/theria/heat_pump_monitor.py`, configured under `options.heat_pump`
in `config.yaml`. Instantaneous power comes from `compressor_speed_2`; COP is computed as a
rolling delta of the delivered/consumed cumulative counters between polls, not from the
counters directly.

---

## Zone Configurations

**Salon temperature points (measured 2026-11-01):** the two thermostats' sensors
(`sensor.vantsidan_temperature`, `sensor.klippsidan_temperature`), the Verisure water alarm on the floor
(`sensor.schamponeringstolen_temperature`) and the Verisure smoke detector in the ceiling
(`sensor.butik_temperature_2`, probably - to be confirmed). They report every few minutes to a few hours. The
Netatmo thermostats stand in `auto` with a schedule (`select.salong` = "Mitt schema"); `switch.vattenavstangning_salongen`
is the salon's water shut-off and is never touched. Wanted: the mean of all points about 19 C weekdays 09-19 and
weekends 10-16, cooler outside those hours; a schedule editable in Theria.

Salon schedule "Mitt schema" (selected), read from the Netatmo API (`homesdata`) 2026-11-08, local time, the same for both rooms.
Home Assistant's Netatmo diagnostics redact the timetable; `NETATMO_CLIENT_ID`, `NETATMO_CLIENT_SECRET` and `NETATMO_REFRESH_TOKEN`
in `.env` give read-only access (scope `read_thermostat`). A manual setpoint lasts at most 8 h (`therm_setpoint_default_duration` 480 min).

| Day | Setpoints (C) |
|---|---|
| Monday-Friday | 04-06: 23 (Komfort +), 06-19: 21 (Komfort), 19-04: 19 (Natt) |
| Saturday | 05-07: 23, 07-18: 21, from 18: 19 |
| Sunday | 08-10: 21, 10-17: 16 (Ekoläge), 17-19: 21, from 19: 19 |

A second stored schedule, "manuellt", is not selected: Komfort 22 from 04 to 19:30 every day except Sunday (07-17), Natt 19.5.

**Guest house (Tado):** `climate.bedroom`, `hallway`, `kitchen`, `tv_room_gh`, `storage_room`: modes off/auto/heat,
step 0.5, range 5-25 C, `hvac_action` available; setpoints 2026-11-01: 18.0 / 19.5 / 20.0 / 19.5 / 17.0. Wanted: a
temperature per room that may vary about +/-1.5 C with the price.

Radiator control is in shadow mode (`core/theria/radiators/`): it logs what it would set, it writes nothing.

### Zone 1: Boutique (32 sqm inside, 2.5 m high, per Johan 2026-11-09)

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
**Source:** Home Assistant's built-in Nord Pool integration, read with the service `nordpool.get_prices_for_date` (15-minute prices, the only price source)
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
