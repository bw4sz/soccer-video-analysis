"""Per-track jersey voting tests on synthetic observation lists."""

from soccer_vision.identify.vote import JerseyVote, vote_jersey


def test_clear_majority_wins():
    obs = [(6, 0.9), (6, 0.8), (6, 0.85), (8, 0.4)]
    v = vote_jersey(obs)
    assert v.jersey == 6
    assert v.confidence > 0.5
    assert v.n_obs == 4


def test_confidence_outweighs_raw_count():
    # 8 appears more often but with weak confidence; 6 is read confidently.
    obs = [(8, 0.2), (8, 0.2), (8, 0.2), (6, 0.95), (6, 0.95)]
    assert vote_jersey(obs).jersey == 6


def test_too_few_votes_is_unknown():
    assert vote_jersey([(6, 0.9), (6, 0.9)], min_votes=3).jersey is None


def test_split_vote_is_unknown():
    # 6 and 8 essentially tie — margin guard should reject.
    obs = [(6, 0.8), (6, 0.8), (8, 0.8), (8, 0.8)]
    assert vote_jersey(obs).jersey is None


def test_plurality_without_majority_is_unknown():
    # 6 leads but holds < 50% of weight across many candidates.
    obs = [(6, 0.5), (6, 0.5), (8, 0.4), (9, 0.4), (10, 0.4)]
    v = vote_jersey(obs, min_share=0.5)
    assert v.jersey is None


def test_empty_observations():
    v = vote_jersey([])
    assert v == JerseyVote(None, 0.0, 0, 0.0)


def test_legible_fraction_reported():
    obs = [(6, 0.9)] * 4
    v = vote_jersey(obs, n_sampled=10)
    assert v.n_obs == 4
    assert abs(v.legible_frac - 0.4) < 1e-9
