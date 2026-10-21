from datetime import datetime, timezone

from app.ingestion.data_pipeline import (
    _extract_line,
    _main_line,
    _parse_game,
    _parse_game_status,
)


def _values(pairs):
    return [{"value": v, "odd": o} for v, o in pairs]


def test_main_line_is_the_most_even_priced_line():
    # Shape taken from a real API-Sports /odds response.
    values = _values([
        ("Over 231", "1.71"), ("Under 231", "2.19"),
        ("Over 232", "1.79"), ("Under 232", "2.09"),
        ("Over 234", "1.98"), ("Under 234", "1.86"),
        ("Over 236", "2.30"), ("Under 236", "1.62"),
    ])
    assert _main_line(values) == 234


def test_main_line_ignores_one_sided_alternates():
    values = _values([("Over 226.5", "1.38"), ("Over 230.5", "1.51"), ("Under 230.5", "2.21")])
    assert _main_line(values) == 230.5
    assert _main_line(_values([("Over 226.5", "1.38")])) is None


def test_extract_line_takes_median_across_bookmakers():
    def book(book_id, bet_id, name, line):
        return {
            "id": book_id,
            "bets": [{"id": bet_id, "name": name, "values": _values([
                (f"Over {line}", "1.91"), (f"Under {line}", "1.91"),
            ])}],
        }

    odds = [{"bookmakers": [
        book(4, 4, "Over/Under", 228.5),
        book(12, 4, "Over/Under", 231.0),
        book(26, 4, "Over/Under", 234.5),
        book(26, 5, "Over/Under 1st Half", 117.5),
    ]}]
    assert _extract_line(odds, (4, "Over/Under")) == 231.0
    assert _extract_line(odds, (5, "Over/Under 1st Half")) == 117.5
    assert _extract_line([], (4, "Over/Under")) is None


def test_status_mapping():
    assert _parse_game_status({"short": "NS"}) == "scheduled"
    assert _parse_game_status({"short": "Q3"}) == "live"
    assert _parse_game_status({"short": "AOT"}) == "finished"
    assert _parse_game_status({"short": "POST"}) == "postponed"
    assert _parse_game_status({"short": "CANC"}) == "cancelled"
    assert _parse_game_status({"short": None}) == "scheduled"


def test_parse_game_stores_tip_off_in_utc():
    raw = {
        "id": 519536,
        "date": "2026-10-21T22:30:00-04:00",
        "status": {"short": "NS"},
        "league": {"id": 12, "season": "2026-2027"},
        "teams": {"home": {"id": 147, "name": "LA Clippers"}, "away": {"id": 159, "name": "Kings"}},
        "scores": {"home": {"total": None}, "away": {"total": None}},
    }
    parsed = _parse_game(raw)
    assert parsed["date"] == datetime(2026, 10, 22, 2, 30, tzinfo=timezone.utc)
    assert parsed["season"] == "2026-2027"
    assert parsed["home_score"] is None and parsed["home_halftime_score"] is None
