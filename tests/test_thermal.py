import pytest

from pest_node.thermal import (SimulatedSensor, ThrottleFlags, parse_get_throttled,
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



def test_simulated_sensor_reads():
    sensor = SimulatedSensor([(50.0, 0x50005)], loop=False)
    sample = sensor.read()
    assert sample.temp_c == 50.0 and sample.flags.under_voltage_now
