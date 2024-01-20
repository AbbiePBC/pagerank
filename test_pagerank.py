from pagerank import transition_model, sample_pagerank


def test_transition_model():

    corpus = {"1.html": {"2.html", "3.html"}, "2.html": {"3.html"}, "3.html": {"2.html"}}
    page = "1.html"
    damping_factor = 0.85
    result = transition_model(corpus, page, damping_factor)

    assert result == {"1.html": 0.05, "2.html": 0.475, "3.html": 0.475}


# the actual results will depend on the random number choice to start with
def test_sample_page_rank():

    corpus = {"1.html": {"2.html", "3.html"}, "2.html": {"3.html"}, "3.html": {"2.html"}}
    damping_factor = 0.85
    n = 10000
    result = sample_pagerank(corpus, damping_factor, n)

    assert result == {"1.html": 0.05, "2.html": 0.475, "3.html": 0.475}




#$ python pagerank.py corpus0
# PageRank Results from Sampling (n = 10000)
#   1.html: 0.2223
#   2.html: 0.4303
#   3.html: 0.2145
#   4.html: 0.1329
# PageRank Results from Iteration
#   1.html: 0.2202
#   2.html: 0.4289
#   3.html: 0.2202
#   4.html: 0.1307
