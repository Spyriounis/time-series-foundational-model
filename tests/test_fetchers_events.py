import pytest

from helpers import utc
from tsfm.constants import TIME_COLUMN
from tsfm.data.fetchers.base import CandleFetcher
from tsfm.data.fetchers.events import CentralBankMeetingFetcher, _parse_meetings
from tsfm.data.fetchers.indices import INDEX_FETCHERS
from tsfm.data.fetchers.registry import build_fetchers
from tsfm.data.storage import DataKey
from tsfm.enums import Dataset, Granularity, PullTarget, Source

MEETINGS_CSV = """date,time_utc,bank,currency,source_url
2024-01-31,19:00,FED,USD,https://example.com/fomc
2024-03-19,,BOJ,JPY,https://example.com/boj
"""


def test_meetings_parse_with_and_without_a_time():
    """Some banks publish no announcement time; such a meeting is kept, at midnight UTC."""
    assert _parse_meetings(MEETINGS_CSV).to_pylist() == [
        {TIME_COLUMN: utc(2024, 1, 31, 19), "event_id": "RATE_DECISION_FED", "currency": "USD"},
        {TIME_COLUMN: utc(2024, 3, 19), "event_id": "RATE_DECISION_BOJ", "currency": "JPY"},
    ]


def test_meetings_live_in_one_events_table(settings):
    """All banks share one table, told apart by `event_id`."""
    assert CentralBankMeetingFetcher(settings).list_keys() == [DataKey(Dataset.EVENTS, (Source.CENTRAL_BANKS,))]


@pytest.mark.parametrize("target", PullTarget)
def test_every_target_builds_fetchers(settings, target):
    """A name the command line accepts must always map to at least one fetcher."""
    assert build_fetchers(target, settings)


def test_indices_target_groups_every_index_file(settings):
    """`indices` is the one target that stands for several fetchers."""
    fetchers = build_fetchers(PullTarget.INDICES, settings)
    assert [type(fetcher) for fetcher in fetchers] == list(INDEX_FETCHERS)


def test_granularities_reach_the_candle_fetchers(settings):
    """`--granularity H1` must narrow the candle pulls, not be dropped on the way."""
    for target in (PullTarget.OANDA_FX, PullTarget.OANDA_OTHER, PullTarget.ETORO):
        (fetcher,) = build_fetchers(target, settings, [Granularity.H1])
        assert isinstance(fetcher, CandleFetcher)
        assert fetcher.granularities == (Granularity.H1,)
