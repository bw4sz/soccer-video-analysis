"""OCR's veto over a re-id name: when it fires, and when it stays quiet."""

from soccer_vision.cli.identify import _apply_crosscheck, _blank, _crosscheck_kwargs
from soccer_vision.identify.crosscheck import AGREE, CONFLICT, NO_EVIDENCE, crosscheck_jersey
from soccer_vision.identify.vote import vote_jersey


def strong(number, n=5, conf=0.9):
    return [(number, conf)] * n


def test_confident_disagreement_is_a_conflict():
    cc = crosscheck_jersey(21, strong(7))
    assert cc.verdict == CONFLICT
    assert cc.jersey == 7
    assert cc.n_obs == 5


def test_confident_agreement_corroborates():
    assert crosscheck_jersey(7, strong(7)).verdict == AGREE


def test_low_confidence_reads_cannot_veto():
    # The number is read repeatedly, but never confidently — exactly the
    # hallucination pattern, so re-id keeps its name.
    cc = crosscheck_jersey(21, [(7, 0.3)] * 10)
    assert cc.verdict == NO_EVIDENCE
    assert cc.n_obs == 0


def test_too_few_strong_reads_cannot_veto():
    assert crosscheck_jersey(21, strong(7, n=3), min_reads=4).verdict == NO_EVIDENCE


def test_split_strong_reads_cannot_veto():
    # Half say 7, half say 9 — the reader is unreliable here, so it gets no vote.
    assert crosscheck_jersey(21, strong(7, n=4) + strong(9, n=4)).verdict == NO_EVIDENCE


def test_a_few_stray_reads_do_not_block_a_veto():
    cc = crosscheck_jersey(21, strong(7, n=8) + [(9, 0.8)])
    assert cc.verdict == CONFLICT and cc.jersey == 7


def test_excluded_numbers_never_veto():
    cc = crosscheck_jersey(21, strong(1), exclude=[1])
    assert cc.verdict == NO_EVIDENCE


def test_no_expected_number_means_nothing_to_contradict():
    # A re-id name with no roster number behind it — the read is recorded but
    # cannot conflict with anything.
    cc = crosscheck_jersey(None, strong(7))
    assert cc.verdict == NO_EVIDENCE
    assert cc.jersey == 7


def test_veto_survives_an_ocr_vote_that_itself_abstained():
    # The naming vote is dragged under its share guard by weak noise reads, yet
    # the high-confidence subset is unanimous — the veto still fires.
    reads = strong(7, n=4) + [(n, 0.25) for n in (1, 2, 3, 4, 5)] * 4
    assert vote_jersey(reads).jersey is None
    assert crosscheck_jersey(21, reads).verdict == CONFLICT


class FakeArgs:
    conflict_min_reads = None
    conflict_min_read_conf = None
    conflict_min_share = None
    conflict_exclude_jersey = None


def reid_result(name="Morgan", jersey=21):
    r = _blank()
    r.update(name=name, jersey=jersey, source="reid", confidence=0.71, similarity=0.71)
    return r


def test_conflicting_track_is_dropped_but_stays_auditable():
    r = reid_result()
    verdict = _apply_crosscheck(41, r, vote_jersey(strong(7)), _crosscheck_kwargs(FakeArgs, {}))

    assert verdict == CONFLICT
    assert r["name"] is None and r["jersey"] is None and r["source"] is None
    assert r["crosscheck"] == CONFLICT
    assert r["conflict"]["reid_name"] == "Morgan"
    assert r["conflict"]["reid_jersey"] == 21
    assert r["conflict"]["ocr_jersey"] == 7
    # The re-id evidence is preserved so the drop can be reviewed.
    assert r["similarity"] == 0.71


def test_corroborated_track_keeps_its_reid_name():
    r = reid_result()
    _apply_crosscheck(41, r, vote_jersey(strong(21)), _crosscheck_kwargs(FakeArgs, {}))

    assert r["name"] == "Morgan" and r["jersey"] == 21 and r["source"] == "reid"
    assert r["crosscheck"] == AGREE


def test_unreadable_track_keeps_its_reid_name():
    r = reid_result()
    _apply_crosscheck(41, r, vote_jersey([]), _crosscheck_kwargs(FakeArgs, {}))

    assert r["name"] == "Morgan" and r["source"] == "reid"
    assert r["crosscheck"] == NO_EVIDENCE
    assert r["conflict"] is None


def linked_chain(*tids):
    """A LinkResult joining every id into one chain, as `link-tracks` would."""
    from soccer_vision.tracking.link import LinkResult

    return LinkResult(parent={t: tids[0] for t in tids}, links=[])


def test_linking_cannot_hand_back_a_vetoed_identity():
    from soccer_vision.tracking.link import propagate_names

    jerseys = {"tracks": {
        "1": {"name": "Morgan", "jersey": 21, "similarity": 0.7, "source": "reid"},
        # Lane 2 was dropped: its shirt reads 7, so it is not Morgan, even though
        # motion says it continues lane 1.
        "2": {"name": None, "jersey": None, "similarity": 0.7,
              "conflict": {"reid_name": "Morgan", "reid_jersey": 21, "ocr_jersey": 7}},
        "3": {"name": None, "jersey": None},
    }}
    doc, stats = propagate_names(jerseys, linked_chain("1", "2", "3"))

    assert doc["tracks"]["3"]["name"] == "Morgan"   # ordinary inheritance
    assert doc["tracks"]["2"]["name"] is None       # veto holds
    assert stats["blocked_by_jersey"] == 1


def test_linking_still_fills_a_lane_whose_reads_agree():
    from soccer_vision.tracking.link import propagate_names

    jerseys = {"tracks": {
        "1": {"name": "Morgan", "jersey": 21, "similarity": 0.7, "source": "reid"},
        "2": {"name": None, "jersey": None,
              "conflict": {"reid_name": "Leire", "reid_jersey": 8, "ocr_jersey": 21}},
    }}
    doc, stats = propagate_names(jerseys, linked_chain("1", "2"))

    # The reads that vetoed "Leire" say 21 — which is exactly what the chain offers.
    assert doc["tracks"]["2"]["name"] == "Morgan"
    assert stats["blocked_by_jersey"] == 0


def test_thresholds_come_from_flags_then_profile_then_default():
    class Args(FakeArgs):
        conflict_min_reads = 6

    kw = _crosscheck_kwargs(Args, {"conflict_min_reads": 9, "conflict_min_read_conf": 0.5})
    assert kw["min_reads"] == 6           # flag wins
    assert kw["min_read_conf"] == 0.5     # profile fills in
    assert kw["min_share"] == 0.75        # built-in default
