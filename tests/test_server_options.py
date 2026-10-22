from app.__main__ import server_options


def test_proxy_headers_are_ignored_without_trusted_proxies(settings):
    options = server_options(settings.model_copy(update={"trusted_proxies": []}))
    assert options["proxy_headers"] is False


def test_trusted_proxies_enable_forwarded_ip(settings):
    options = server_options(settings.model_copy(update={"trusted_proxies": ["127.0.0.1", "172.18.0.1"]}))
    assert options["proxy_headers"] is True
    assert options["forwarded_allow_ips"] == "127.0.0.1,172.18.0.1"
