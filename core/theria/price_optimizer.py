"""
Electricity Price Optimization

Fetches Nordpool prices and uses hybrid classification (PumpSteer-inspired):
- Percentile-based categorization (5 levels)
- Historical context (7-day rolling average)
- Extreme price detection (>3x average)
"""

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PriceOptimizer:
    """Hybrid price-based temperature optimization (PumpSteer-inspired)."""

    # Price categories
    VERY_CHEAP = "VERY_CHEAP"
    CHEAP = "CHEAP"
    NORMAL = "NORMAL"
    EXPENSIVE = "EXPENSIVE"
    VERY_EXPENSIVE = "VERY_EXPENSIVE"
    EXTREME_EXPENSIVE = "EXTREME_EXPENSIVE"

    def __init__(
        self,
        ha_client,
        price_entity: str,
        expensive_hours: int = 4,  # Kept for backward compatibility
        cheap_hours: int = 4,      # Kept for backward compatibility
        adjustment_degrees: float = 0.5
    ):
        """Initialize price optimizer.

        Args:
            ha_client: Home Assistant API client
            price_entity: Entity ID for Nordpool price sensor
            expensive_hours: Legacy parameter (kept for compatibility)
            cheap_hours: Legacy parameter (kept for compatibility)
            adjustment_degrees: Base temperature adjustment in °C
        """
        self.ha_client = ha_client
        self.price_entity = price_entity
        self.adjustment_degrees = adjustment_degrees

        # Price forecast cache
        self.price_forecast = []
        self.last_update = None

        # Historical price tracking (7 days × 24 hours = 168 prices)
        self.historical_prices = deque(maxlen=7 * 24)

        # Legacy sets (kept for backward compatibility with API)
        self.expensive_hour_set = set()
        self.cheap_hour_set = set()

        # Price category mapping for current forecast
        self.hour_categories = {}  # {hour: category}

    def update_prices(self):
        """Fetch latest price forecast from Home Assistant."""
        try:
            state = self.ha_client.get_state(self.price_entity)

            # Nordpool sensor stores hourly prices in attributes
            raw_today = state.get("attributes", {}).get("raw_today", [])
            raw_tomorrow = state.get("attributes", {}).get("raw_tomorrow", [])

            # Combine today and tomorrow prices
            all_prices = raw_today + raw_tomorrow

            if not all_prices:
                logger.warning(f"No price data available from {self.price_entity}")
                return

            # Parse price forecast (list of dicts with 'start' and 'value' or 'price')
            self.price_forecast = []
            for entry in all_prices:
                try:
                    start_time = datetime.fromisoformat(entry["start"].replace("Z", "+00:00"))
                    # Handle both 'value' (official integration) and 'price' (custom integrations)
                    price = float(entry.get("value") or entry.get("price"))
                    self.price_forecast.append({
                        "hour": start_time.hour,
                        "date": start_time.date(),
                        "timestamp": start_time,
                        "price": price
                    })
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse price entry: {entry}, error: {e}")
                    continue

            # Add to historical tracking
            for entry in self.price_forecast:
                self.historical_prices.append(entry["price"])

            # Classify prices using hybrid method
            self._classify_prices_hybrid()

            # Update legacy sets for backward compatibility
            self._update_legacy_sets()

            self.last_update = datetime.now()

            # Log price distribution
            category_counts = {}
            for category in self.hour_categories.values():
                category_counts[category] = category_counts.get(category, 0) + 1

            logger.info(
                f"Updated prices: {len(self.price_forecast)} hours, "
                f"categories: {category_counts}, "
                f"historical avg: {np.mean(self.historical_prices):.2f} (7d)" if self.historical_prices else "no history"
            )

        except Exception as e:
            logger.error(f"Failed to update prices: {e}", exc_info=True)

    def _classify_prices_hybrid(self):
        """Classify prices using hybrid approach (PumpSteer-inspired).

        Combines:
        1. Percentile-based classification (relative to 24h forecast)
        2. Historical context (7-day rolling average)
        3. Extreme price detection (>3x historical average)
        """
        if not self.price_forecast:
            return

        # Extract prices for percentile calculation
        prices_24h = [entry["price"] for entry in self.price_forecast]

        # Calculate percentiles (20%, 40%, 60%, 80%)
        p20 = np.percentile(prices_24h, 20)
        p40 = np.percentile(prices_24h, 40)
        p60 = np.percentile(prices_24h, 60)
        p80 = np.percentile(prices_24h, 80)

        # Historical average (trailing 7 days)
        historical_avg = np.mean(self.historical_prices) if len(self.historical_prices) > 24 else None

        # Classify each hour
        self.hour_categories = {}
        for entry in self.price_forecast:
            price = entry["price"]
            hour = entry["hour"]

            # Check for extreme prices first (if we have historical data)
            if historical_avg and price > 3 * historical_avg:
                category = self.EXTREME_EXPENSIVE
            # Percentile-based classification
            elif price < p20:
                category = self.VERY_CHEAP
            elif price < p40:
                category = self.CHEAP
            elif price < p60:
                category = self.NORMAL
            elif price < p80:
                category = self.EXPENSIVE
            else:
                category = self.VERY_EXPENSIVE

            self.hour_categories[hour] = category

    def _update_legacy_sets(self):
        """Update legacy expensive_hour_set and cheap_hour_set for backward compatibility."""
        self.cheap_hour_set = {
            hour for hour, cat in self.hour_categories.items()
            if cat in (self.VERY_CHEAP, self.CHEAP)
        }
        self.expensive_hour_set = {
            hour for hour, cat in self.hour_categories.items()
            if cat in (self.EXPENSIVE, self.VERY_EXPENSIVE, self.EXTREME_EXPENSIVE)
        }

    def get_price_adjustment(self, current_time: Optional[datetime] = None) -> float:
        """Get temperature adjustment based on current price category.

        Args:
            current_time: Time to check (defaults to now)

        Returns:
            Temperature adjustment in °C (scaled by category)
        """
        if current_time is None:
            current_time = datetime.now()

        current_hour = current_time.hour

        # Check if we need to update prices (every hour)
        if (self.last_update is None or
            (datetime.now() - self.last_update) > timedelta(hours=1)):
            self.update_prices()

        # Get category for current hour
        category = self.hour_categories.get(current_hour, self.NORMAL)

        # Apply scaled adjustment based on category
        # More extreme categories = larger adjustments
        if category == self.EXTREME_EXPENSIVE:
            return -2.0 * self.adjustment_degrees  # -1.0°C (emergency reduction)
        elif category == self.VERY_EXPENSIVE:
            return -1.5 * self.adjustment_degrees  # -0.75°C
        elif category == self.EXPENSIVE:
            return -1.0 * self.adjustment_degrees  # -0.5°C
        elif category == self.NORMAL:
            return 0.0  # No adjustment
        elif category == self.CHEAP:
            return 1.0 * self.adjustment_degrees   # +0.5°C
        elif category == self.VERY_CHEAP:
            return 1.5 * self.adjustment_degrees   # +0.75°C
        else:
            return 0.0

    def get_current_price(self) -> Optional[float]:
        """Get current electricity price."""
        current_hour = datetime.now().hour
        for entry in self.price_forecast:
            if entry["hour"] == current_hour and entry["date"] == datetime.now().date():
                return entry["price"]
        return None

    def get_current_category(self, current_time: Optional[datetime] = None) -> str:
        """Get price category for current hour.

        Args:
            current_time: Time to check (defaults to now)

        Returns:
            Price category string (VERY_CHEAP, CHEAP, NORMAL, EXPENSIVE, VERY_EXPENSIVE, EXTREME_EXPENSIVE)
        """
        if current_time is None:
            current_time = datetime.now()

        current_hour = current_time.hour

        # Update prices if needed
        if (self.last_update is None or
            (datetime.now() - self.last_update) > timedelta(hours=1)):
            self.update_prices()

        return self.hour_categories.get(current_hour, self.NORMAL)

    def get_price_forecast_24h(self) -> list[dict]:
        """Get next 24 hours of price forecast."""
        now = datetime.now()
        forecast_24h = []

        for entry in self.price_forecast:
            if entry["timestamp"] >= now and entry["timestamp"] < now + timedelta(hours=24):
                forecast_24h.append(entry)

        return sorted(forecast_24h, key=lambda x: x["timestamp"])
