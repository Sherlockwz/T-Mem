"""DateTime utilities for T_mem (timezone-aware helpers, ms/s timestamp converters)."""

import datetime
from zoneinfo import ZoneInfo
import os
import logging

logger = logging.getLogger(__name__)


def get_timezone() -> ZoneInfo:
    tz = os.getenv("TZ", "Asia/Shanghai")
    return ZoneInfo(tz)


timezone = get_timezone()


def get_now_with_timezone() -> datetime.datetime:
    return datetime.datetime.now(tz=timezone)


def to_timezone(dt: datetime.datetime, tz: ZoneInfo = None) -> datetime.datetime:
    if tz is None:
        tz = timezone
    return dt.astimezone(tz)


def to_iso_format(dt: datetime.datetime) -> str:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone)
    return dt.astimezone(timezone).isoformat()


def from_timestamp(timestamp: int | float) -> datetime.datetime:
    if timestamp >= 1e12:
        timestamp_seconds = timestamp / 1000.0
    else:
        timestamp_seconds = timestamp

    return datetime.datetime.fromtimestamp(timestamp_seconds, tz=timezone)


def to_timestamp(dt: datetime.datetime) -> int:
    return int(dt.timestamp())


def to_timestamp_ms(dt: datetime.datetime) -> int:
    return int(dt.timestamp() * 1000)


def to_timestamp_ms_universal(time_value) -> int:
    """Universal time-value -> ms timestamp. Accepts int/float/str/datetime/None."""
    try:
        if time_value is None:
            return 0

        if isinstance(time_value, (int, float)):
            if time_value >= 1e12:
                return int(time_value)
            else:
                return int(time_value * 1000)

        if isinstance(time_value, str):
            try:
                numeric_value = float(time_value)
                return to_timestamp_ms_universal(numeric_value)
            except ValueError:
                dt = from_iso_format(time_value)
                return to_timestamp_ms(dt)

        if isinstance(time_value, datetime.datetime):
            return to_timestamp_ms(time_value)

        return to_timestamp_ms_universal(str(time_value))

    except Exception as e:
        logger.error("[DateTimeUtils] to_timestamp_ms_universal - Error converting time value %s: %s", time_value, str(e))
        return 0


def from_iso_format(create_time, target_timezone: ZoneInfo = None) -> datetime.datetime:
    """ISO string / datetime -> tz-aware datetime; falls back to now() on failure."""
    try:
        if isinstance(create_time, datetime.datetime):
            dt = create_time
        elif isinstance(create_time, str):
            dt = datetime.datetime.fromisoformat(create_time)
        else:
            dt = datetime.datetime.fromisoformat(str(create_time))

        if dt.tzinfo is None:
            tz = target_timezone or get_timezone()
            dt_localized = dt.replace(tzinfo=tz)
        else:
            dt_localized = dt

        return dt_localized.astimezone(get_timezone())

    except Exception as e:
        logger.error("[DateTimeUtils] from_iso_format - Error converting time: %s", str(e))
        return get_now_with_timezone()
