import pytest

from pest_node.thermal import (AdaptiveController, SimulatedSensor, ThrottleFlags, parse_get_throttled,
                               parse_measure_clock, parse_measure_temp)


def test_parse_vcgencmd_outputs():
    # saídas reais do Pi do projeto
    assert parse_measure_temp("temp=53.7'C\n") == 53.7
    assert parse_get_throttled("throttled=0x50005\n") == 0x50005
    assert parse_measure_clock("frequency(48)=600062000\n") == pytest.approx(600.062)
    assert parse_measure_temp("lixo") is None
    assert parse_get_throttled("") is None


def test_flags_of_real_pi():
    f = ThrottleFlags(0x50005)
    assert f.under_voltage_now and f.throttled_now
    assert not f.freq_capped_now and not f.soft_temp_limit_now
    assert f.under_voltage_occurred and f.throttled_occurred
    assert f.active() == ["under_voltage", "throttled"]
    assert "0x50005" in f.describe()


def _ctrl(**kw):
    defaults = dict(base_interval_s=1.0, temp_high_c=70, temp_resume_c=65, slowdown_factor=2, max_interval_s=8)
    defaults.update(kw)
    return AdaptiveController(**defaults)


def run(ctrl, readings):
    sensor = SimulatedSensor(readings, loop=False)
    return [ctrl.update(sensor.read()) for _ in readings]


def test_heats_up_backs_off_then_recovers_with_hysteresis():
    ctrl = _ctrl()
    intervals = run(ctrl, [(50, 0), (71, 0), (72, 0), (75, 0), (80, 0), (80, 0),
                           (68, 0),  # entre limiares: mantém
                           (64, 0), (60, 0), (55, 0), (50, 0)])
    assert intervals == [1, 2, 4, 8, 8, 8, 8, 4, 2, 1, 1]
    assert not ctrl.degraded and ctrl.reason == "normal"


def test_throttling_bit_triggers_even_when_cool():
    ctrl = _ctrl()
    assert run(ctrl, [(50, 0x4)]) == [2]
    assert ctrl.reason == "throttling"


def test_under_voltage_configurable():
    assert run(_ctrl(), [(50, 0x1)]) == [2]
    assert run(_ctrl(react_to_under_voltage=False, react_to_throttling=False), [(50, 0x50005)]) == [1]


def test_past_bits_do_not_trigger():
    assert run(_ctrl(), [(50, 0x50000)]) == [1]


def test_missing_sensor_data_is_safe():
    ctrl = _ctrl()
    assert run(ctrl, [(None, None), (None, None)]) == [1, 1]


def test_degraded_flag():
    ctrl = _ctrl()
    run(ctrl, [(75, 0)])
    assert ctrl.degraded and "temp" in ctrl.reason


def test_capture_interval_above_max_still_flags_degraded():
    ctrl = _ctrl(base_interval_s=10, max_interval_s=8)
    assert run(ctrl, [(75, 0)]) == [10]  # já está no teto, mas está sob pressão
    assert ctrl.degraded
    assert run(ctrl, [(50, 0)]) == [10] and not ctrl.degraded


def test_set_base_interval_keeps_thermal_factor():
    ctrl = _ctrl(base_interval_s=1, max_interval_s=60)
    run(ctrl, [(75, 0)])  # 2 s
    ctrl.set_base_interval(5)
    assert ctrl.interval_s == 10 and ctrl.base_interval_s == 5
