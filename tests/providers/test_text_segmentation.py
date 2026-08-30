from hermes_realtime.providers._text_segmentation import first_speakable_sentence_end


def test_short_leading_fragments_are_coalesced_with_following_sentence() -> None:
    for text in ("No. I can't switch models.", "#. Continue with the answer."):
        assert first_speakable_sentence_end(text) == len(text)


def test_incomplete_short_fragment_remains_buffered() -> None:
    assert first_speakable_sentence_end("No. ") is None
    assert first_speakable_sentence_end("#. ") is None


def test_substantive_short_sentence_still_streams_immediately() -> None:
    text = "Yes. Continue with the answer."
    assert first_speakable_sentence_end(text) == len("Yes.")
