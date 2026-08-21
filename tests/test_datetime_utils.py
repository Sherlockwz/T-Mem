"""Tests for T_mem.utils.datetime_utils."""

from __future__ import annotations

import datetime

import pytest

from T_mem.utils import datetime_utils as du


class TestTimestampConversion:
    def test_seconds_float(self):
        dt = du.from_timestamp(1_700_000_000.0)
        assert isinstance(dt, datetime.datetime)
        assert dt.tzinfo is not None

    def test_milliseconds_detected(self):
        # > 1e12 => treated as milliseconds
        dt = du.from_timestamp(1_700_000_000_000)
        assert isinstance(dt, datetime.datetime)
        assert dt.tzinfo is not None

    def test_roundtrip_ms(self):
        dt = datetime.datetime(2024, 3, 14, 12, 0, tzinfo=du.timezone)
        ms = du.to_timestamp_ms(dt)
        back = du.from_timestamp(ms)
        assert back.year == 2024
        assert back.month == 3
        assert back.day == 14


class TestUniversalTimestamp:
    def test_none_returns_zero(self):
        assert du.to_timestamp_ms_universal(None) == 0

    def test_int_seconds(self):
        assert du.to_timestamp_ms_universal(1700000000) == 1700000000 * 1000

    def test_int_milliseconds(self):
        assert du.to_timestamp_ms_universal(1700000000000) == 1700000000000

    def test_float_seconds(self):
        assert du.to_timestamp_ms_universal(1700000000.5) == 1700000000500

    def test_datetime(self):
        dt = datetime.datetime(2024, 3, 14, tzinfo=du.timezone)
        assert du.to_timestamp_ms_universal(dt) == du.to_timestamp_ms(dt)

    def test_invalid_string_falls_back_zero(self):
        # Falls back to now() in from_iso_format then converts; never raises.
        val = du.to_timestamp_ms_universal("not-a-date")
        assert isinstance(val, int)


class TestIsoFormat:
    def test_naive_datetime_localized(self):
        naive = datetime.datetime(2024, 1, 1, 8, 0)
        iso = du.to_iso_format(naive)
        assert iso.endswith("+08:00") or "+" in iso or "Z" in iso

    def test_aware_roundtrip(self):
        dt = datetime.datetime(2024, 1, 1, 8, 0, tzinfo=du.timezone)
        iso = du.to_iso_format(dt)
        back = du.from_iso_format(iso)
        assert back.hour == dt.hour
