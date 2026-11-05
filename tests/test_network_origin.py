import pytest

from hermes_realtime.network_origin import canonical_remote_hostname


@pytest.mark.parametrize("value", ["8.8.8.8 evil", "1.2.3.4\x0bz", "8.8.8.8\t"])
def test_remote_hostname_refuses_whitespace_before_platform_parser(value):
    with pytest.raises(ValueError):
        canonical_remote_hostname(value, boundary="fixture")
