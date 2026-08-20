from datetime import datetime, timedelta

import pandas as pd
import requests


API_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/evenements-publics-openagenda/records"
)

PAGE_SIZE = 100


def fetch_events(city: str = "Metz") -> pd.DataFrame:
    """
    Récupère tous les événements OpenAgenda récents ou à venir
    pour une ville donnée.

    Un événement est conservé si sa dernière date de fin est
    postérieure ou égale à la date située un an avant aujourd'hui.
    """

    one_year_ago = datetime.now() - timedelta(days=365)

    events = []
    offset = 0

    while True:
        params = {
            "where": (
                f'location_city = "{city}" '
                f'AND lastdate_end >= date\'{one_year_ago:%Y-%m-%d}\''
            ),
            "limit": PAGE_SIZE,
            "offset": offset,
        }

        response = requests.get(
            API_URL,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()
        results = data["results"]

        events.extend(results)

        if len(results) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    return pd.DataFrame(events)