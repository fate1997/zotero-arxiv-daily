from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from zotero_arxiv_daily.protocol import Paper


@pytest.mark.parametrize(('response', 'expected'), [
    ('["First University", "Second University", "First University"]', ['First University', 'Second University']),
    ("['First University', 'Second University', 'First University']", ['First University', 'Second University']),
    ('```json\n[" University ", ""]\n```', ['University']),
    ('[]', []),
    ('["University", 123]', None),
    ('[unexpected_function()]', None),
])
def test_provider_affiliation_formats_are_parsed_without_executing_code(monkeypatch, response, expected):
    tokenizer = Mock()
    tokenizer.encode.return_value = [1]
    tokenizer.decode.return_value = 'Paper text'
    monkeypatch.setattr('zotero_arxiv_daily.protocol.tiktoken.encoding_for_model', lambda model: tokenizer)
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=response))])
    paper = Paper('arxiv', 'Paper', [], 'Abstract', 'https://example.org/paper', full_text='Author affiliations')
    assert paper.generate_affiliations(client, {}) == expected
    assert paper.affiliations == expected
