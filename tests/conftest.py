import pytest
import hydra

def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", help="Run tests requiring network, models, or local services")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires network, model downloads, or local services")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        skip = pytest.mark.skip(reason="Use --run-integration for network/service tests")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def config():
    with hydra.initialize(config_path='../config',version_base=None):
        config = hydra.compose(config_name="default")
    config.email.smtp_server = "localhost"
    config.email.smtp_port = 1025
    config.email.smtp_security = "plain"
    config.email.smtp_auth = False
    config.email.sender = "test@example.com"
    config.email.receiver = "test@example.com"
    config.email.sender_password = "test"
    config.llm.api.base_url = "http://localhost:30000/v1"
    config.llm.api.key = "sk-xxx"
    config.reranker.api.base_url = "http://localhost:30000/v1"
    config.reranker.api.key = "sk-xxx"
    config.reranker.api.model = "text-embedding-3-large"
    config.zotero.user_id = "0"
    config.zotero.api_key = "test-zotero-key"
    return config

