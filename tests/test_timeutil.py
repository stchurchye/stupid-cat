from datetime import datetime, timezone

from stupid_cat.timeutil import in_time_range, local_date_key, local_iso_cutoff, parse_iso


def test_in_time_range_handles_mixed_offsets() -> None:
    visit = "2026-06-07T23:00:00+08:00"
    # Same instant as 2026-06-07T15:00:00+00:00 — should be excluded by UTC cutoff after that.
    cutoff_utc = "2026-06-07T16:00:00+00:00"
    assert in_time_range(visit, from_ts=cutoff_utc) is False

    cutoff_local = "2026-06-07T15:00:00+08:00"
    assert in_time_range(visit, from_ts=cutoff_local) is True


def test_local_date_key_uses_local_calendar_day() -> None:
    assert local_date_key("2026-06-14T22:12:16+08:00") == "2026-06-14"


def test_local_iso_cutoff_is_timezone_aware() -> None:
    cutoff = local_iso_cutoff(days=1)
    dt = parse_iso(cutoff)
    assert dt.tzinfo is not None
    assert dt <= datetime.now(timezone.utc).astimezone()
