from src.context.trend_engine import compute_slope, classify_slope


def test_rising_fast():
    times = [0, 60, 120]  # minutes
    values = [85, 95, 120]  # heart rate rising fast
    slope = compute_slope(times, values)
    label = classify_slope(slope)
    assert label == "rising_fast", f"Expected rising_fast, got {label}"
    print(f"PASS: slope={slope:.2f}, label={label}")


def test_stable():
    times = [0, 60, 120]
    values = [85, 86, 85]
    slope = compute_slope(times, values)
    assert classify_slope(slope) == "stable"
    print("PASS: stable trend")
