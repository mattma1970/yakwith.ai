import utils.text_processing as text_processing


def test_regex_filter():
    assert (
        text_processing.remove_problem_chars('A #test,"#!%^*" of.', "[^a-zA-Z,. ]")
        == "A test, of."
    )


def test_regex_filter_default():
    assert text_processing.remove_problem_chars('A #test,"#!%^*" of.') == "A test, of."
