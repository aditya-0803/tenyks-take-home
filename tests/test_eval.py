import pytest

from kiosk_analytics.analytics import PersonResult
from kiosk_analytics.eval import GtPerson, evaluate, temporal_iou


def pred(pid, segments, engaged=True):
    dwell = sum(e - s for s, e in segments)
    return PersonResult(
        pid=pid, dwell_s=dwell, engaged=engaged, segments=segments,
        first_seen=segments[0][0], last_seen=segments[-1][1],
    )


def test_temporal_iou():
    assert temporal_iou([(0, 10)], [(0, 10)]) == pytest.approx(1.0)
    assert temporal_iou([(0, 10)], [(5, 15)]) == pytest.approx(5 / 15)
    assert temporal_iou([(0, 10)], [(20, 30)]) == 0.0


def test_perfect_match():
    p = [pred(1, [(0, 30)]), pred(2, [(40, 100)])]
    g = [GtPerson("P1", [(0, 30)]), GtPerson("P2", [(40, 100)])]
    m = evaluate(p, g)
    assert m["count"]["abs_error"] == 0
    assert m["dwell"]["mae_s"] == pytest.approx(0.0)
    assert m["dwell"]["n_matched"] == 2


def test_dwell_error_and_count_error():
    p = [pred(1, [(0, 25)])]  # 25s vs GT 30s
    g = [GtPerson("P1", [(0, 30)]), GtPerson("P2", [(50, 80)])]  # missed P2
    m = evaluate(p, g)
    assert m["count"]["gt"] == 2 and m["count"]["pred"] == 1
    assert m["count"]["abs_error"] == 1
    assert m["dwell"]["mae_s"] == pytest.approx(5.0)
    assert m["unmatched_gt_persons"] == ["P2"]


def test_non_engaged_predictions_excluded():
    p = [pred(1, [(0, 30)]), pred(2, [(0, 5)], engaged=False)]
    g = [GtPerson("P1", [(0, 30)])]
    m = evaluate(p, g)
    assert m["count"]["pred"] == 1


def test_no_spurious_match_when_disjoint():
    p = [pred(1, [(0, 10)])]
    g = [GtPerson("P1", [(500, 600)])]
    m = evaluate(p, g)
    assert m["dwell"]["n_matched"] == 0
    assert m["unmatched_pred_persons"] == [1]
