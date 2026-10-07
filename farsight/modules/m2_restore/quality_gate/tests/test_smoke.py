from farsight.core import registry


def test_gate_median_threshold():
    g = registry.build("quality_gate", q0=10)
    assert g([1, 2, 30]) is True            # median 2 < 10
    assert g([5, 30, 40]) is False          # median 30
    assert g([10, 10, 10]) is False         # strict <
    assert g([]) is False and g([float("nan")]) is False
    assert g([float("nan"), 1, 2]) is True  # NaN ignored
    assert registry.build("quality_gate").q0 > 0  # config default loads
