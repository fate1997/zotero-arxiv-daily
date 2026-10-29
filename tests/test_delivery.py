from unittest.mock import Mock

import pytest

from zotero_arxiv_daily.utils import send_email


def test_ssl_port_uses_ssl_immediately(config, monkeypatch):
    config.email.smtp_port = 465
    config.email.smtp_security = "auto"
    config.email.smtp_auth = True
    server = Mock()
    server.__enter__ = Mock(return_value=server)
    server.__exit__ = Mock(return_value=False)
    ssl = Mock(return_value=server)
    plain = Mock()
    monkeypatch.setattr("zotero_arxiv_daily.utils.smtplib.SMTP_SSL", ssl)
    monkeypatch.setattr("zotero_arxiv_daily.utils.smtplib.SMTP", plain)
    send_email(config, "<p>Digest</p>")
    ssl.assert_called_once()
    plain.assert_not_called()
    server.starttls.assert_not_called()
    server.login.assert_called_once()
    server.sendmail.assert_called_once()


def test_starttls_failure_never_falls_back_or_sends_credentials(config, monkeypatch):
    config.email.smtp_port = 587
    config.email.smtp_security = "auto"
    config.email.smtp_auth = True
    server = Mock()
    server.__enter__ = Mock(return_value=server)
    server.__exit__ = Mock(return_value=False)
    server.starttls.side_effect = RuntimeError("TLS failure")
    factory = Mock(return_value=server)
    monkeypatch.setattr("zotero_arxiv_daily.utils.smtplib.SMTP", factory)
    with pytest.raises(RuntimeError, match="TLS"):
        send_email(config, "Digest")
    factory.assert_called_once()
    server.login.assert_not_called()
    server.sendmail.assert_not_called()
