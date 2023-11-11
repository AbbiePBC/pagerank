from pagerank import transition_model


def test_transition_model():

    corpus = {"1.html": {"2.html", "3.html"}, "2.html": {"3.html"}, "3.html": {"2.html"}}
    page = "1.html"
    damping_factor = 0.85
    result = transition_model(corpus, page, damping_factor)

    assert result == {"1.html": 0.05, "2.html": 0.475, "3.html": 0.475}
