from zotero_arxiv_daily.reranker.local import LocalReranker
import pytest

@pytest.mark.integration
def test_local_reranker(config):
    reranker = LocalReranker(config)
    score = reranker.get_similarity_score(["hello","world"], ["ping"])
    assert score.shape == (2,1)


@pytest.mark.parametrize("debug", [True, False])
def test_local_embedding_import_works_in_debug_and_regular_mode(config, monkeypatch, debug):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock
    import numpy as np

    config.executor.debug = debug
    encoder = Mock()
    encoder.similarity.return_value.numpy.return_value = np.ones((2, 1))
    constructor = Mock(return_value=encoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor))
    # The quiet mode imports logging helpers; avoid loading the model stack.
    monkeypatch.setitem(sys.modules, "transformers.utils", SimpleNamespace(logging=Mock()))
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", SimpleNamespace(logging=Mock()))
    scores = LocalReranker(config).get_similarity_score(["a", "b"], ["c"])
    assert scores.shape == (2, 1)
    assert encoder.encode.call_count == 2
