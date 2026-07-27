import pytest

from scripts.lib.model_primitives import make_chat_prefix, require_direct_answer_prefix


class RecordingTokenizer:
    chat_template = "template"

    def __init__(self, rendered: str) -> None:
        self.rendered = rendered
        self.kwargs = None

    def apply_chat_template(self, _messages, **kwargs):
        self.kwargs = kwargs
        return self.rendered


def test_candidate_scoring_disables_thinking() -> None:
    tokenizer = RecordingTokenizer("assistant\n<think>\n\n</think>\n")
    prefix = make_chat_prefix(tokenizer, "Question")
    assert prefix.endswith("</think>\n")
    assert tokenizer.kwargs["enable_thinking"] is False


def test_open_thinking_prefix_is_rejected() -> None:
    with pytest.raises(ValueError, match="open <think>"):
        require_direct_answer_prefix("assistant\n<think>\n")
