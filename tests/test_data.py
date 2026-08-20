import pandas as pd
import pytest

from src.data_loader import fetch_events
from src.preprocessing import preprocess_events


@pytest.fixture(scope="session")
def events():
    raw_events = fetch_events(city="Metz")
    return preprocess_events(raw_events)


def test_events_are_retrieved(events):
    assert not events.empty


def test_required_columns_are_present(events):
    required_columns = {
        "uid",
        "title_fr",
        "description_fr",
        "firstdate_begin",
        "lastdate_end",
        "location_city",
    }

    assert required_columns.issubset(events.columns)


def test_events_are_in_metz(events):
    cities = (
        events["location_city"]
        .str.strip()
        .str.lower()
        .unique()
    )

    assert set(cities) == {"metz"}


def test_events_are_recent_or_upcoming(events):
    one_year_ago = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)

    assert (events["lastdate_end"] >= one_year_ago).all()


def test_events_have_titles(events):
    assert events["title_fr"].notna().all()
    assert events["title_fr"].str.strip().ne("").all()


def test_event_uids_are_unique(events):
    assert events["uid"].notna().all()
    assert events["uid"].is_unique