import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor) -> dict[str, float]:
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    probability_of_random_page_selection = (1.0 - damping_factor) / len(corpus.keys())

    probability_dict: dict[str, float] = {key:probability_of_random_page_selection \
                                          for key in corpus.keys()}
    # assumes that all pages are keys in the corpus - i.e. there are no
    # pages in the values that are not keys in the corpus

    pages_linked_to = corpus[page]
    for p in pages_linked_to:
        probability_dict[p] += damping_factor / len(pages_linked_to)

    # remove floating point rounding errors
    for k, p in probability_dict.items():
        probability_dict[k] = round(p, 6)

    return probability_dict


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n`, where `n` > 1, pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    page = random.choice(list(corpus.keys()))
    page_rank_dict: dict[str, int] = {p: 0 for p in corpus.keys()}
    page_rank_dict[page] = 1

    for _ in range(1, n):
        probability_dict = transition_model(corpus, page, damping_factor)
        page = random.choices(list(probability_dict.keys()),
                              weights=list(probability_dict.values()), k=1)[0]
        page_rank_dict[page] += 1


    # now normalise
    for k, v in page_rank_dict.items():
        page_rank_dict[k] = v / n

    return page_rank_dict


def calc_probability(corpus, page, damping_factor, pagerank):

    return (1 - damping_factor) / len(corpus.keys()) + \
                        damping_factor * sum([pagerank[link] / len(corpus[link]) \
                                                for link in corpus[page]])




def iterate_probability(corpus, page, damping_factor, pagerank):

    not_converged = True
    new_pagerank = {p:0.0 for p in pagerank.keys()}
    while not_converged:
        not_converged = False

        for page, links in corpus.items():
            if len(links) == 0:
                new_pagerank[page] = 1 / len(corpus.keys())
            else:
                new_pagerank[page] = calc_probability(corpus, page, damping_factor, new_pagerank)
            if abs(new_pagerank[page] - pagerank[page]) > 0.001:
                not_converged = True

        pagerank = new_pagerank
    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus.keys())
    pagerank = {page:1/num_pages for page in corpus.keys()}

    page = random.choice(list(corpus.keys())) # start at random page

    pagerank = iterate_probability(corpus, page, damping_factor, pagerank)

    total = sum(pagerank.values())
    for p, v in pagerank.items():
        pagerank[p] = round(v / total, 3)

    return pagerank




if __name__ == "__main__":
    main()
