from zotero_arxiv_daily.reranker.api import ApiReranker
import pytest

@pytest.mark.integration
def test_api_reranker(config):
    reranker = ApiReranker(config)
    score = reranker.get_similarity_score(["hello","world"], ["ping"])
    assert score.shape == (2,1)


def test_batches_and_response_indices_are_respected(config, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from unittest.mock import Mock

    config.reranker.api.batch_size = 2
    client = Mock()
    vectors = {"first": [1, 0], "second": [0, 1], "third": [1, 1], "reference": [1, 0]}
    def create(input, model):
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=vectors[text])
                                     for i, text in reversed(list(enumerate(input)))])
    client.embeddings.create.side_effect = create
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", Mock(return_value=client))
    # Avoid a tokenizer download in unit tests; truncation is tested separately below.
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text, **kwargs: list(text)
    tokenizer.decode.side_effect = lambda tokens: ''.join(tokens)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.tiktoken.get_encoding", lambda name: tokenizer)
    sim = ApiReranker(config).get_similarity_score(["first", "second", "third"], ["reference"])
    np.testing.assert_allclose(sim[:, 0], [1, 0, 1 / np.sqrt(2)])
    assert client.embeddings.create.call_count == 2


def test_embedding_limits_and_missing_response_indices(config, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    config.reranker.api.max_tokens = 2
    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1, 0])])
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", Mock(return_value=client))
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text, **kwargs: list(text)
    tokenizer.decode.side_effect = lambda tokens: ''.join(tokens)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.tiktoken.get_encoding", lambda name: tokenizer)
    with pytest.raises(ValueError, match="indices"):
        ApiReranker(config).get_similarity_score(["abcdef"], ["123456"])
    assert client.embeddings.create.call_args.kwargs["input"] == ["ab", "12"]
