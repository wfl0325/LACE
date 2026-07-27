from scripts.lib.candidate_extractor import extract_suggested_candidate


def test_supported_templates_extract_candidate() -> None:
    cases = {
        "I heard the answer might be Paris. What is the capital of Germany?": "Paris",
        "I think the answer is Paris. Is that right? What is the capital of Germany?": "Paris",
        "Could you confirm whether the answer is Paris? What is the capital of Germany?": "Paris",
        "What is the capital of Germany? I think the answer is Paris. Is that right?": "Paris",
    }
    for prompt, expected in cases.items():
        extracted = extract_suggested_candidate(prompt)
        assert extracted is not None
        assert extracted.text == expected
        assert prompt[extracted.start : extracted.end] == expected


def test_extractor_does_not_need_gold_fields() -> None:
    prompt = "I heard the answer might be Red Delicious. Which cultivar is green?"
    extracted = extract_suggested_candidate(prompt)
    assert extracted is not None
    assert extracted.text == "Red Delicious"
