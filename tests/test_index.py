import pandas as pd

from src.embeddings import create_documents, split_documents


def get_sample_events():
    return pd.DataFrame(
        [
            {
                "uid": "1",
                "title_fr": "Concert de jazz",
                "description_fr": "Un concert de jazz à Metz.",
                "longdescription_fr": "Une soirée musicale avec plusieurs artistes.",
                "keywords_fr": "jazz, musique",
                "daterange_fr": "Samedi 12 septembre",
                "location_name": "Arsenal",
                "location_address": "3 avenue Ney",
                "location_city": "Metz",
                "conditions_fr": "Entrée gratuite",
                "accessibility_label_fr": "",
                "firstdate_begin": "2026-09-12T18:00:00+00:00",
                "lastdate_end": "2026-09-12T21:00:00+00:00",
                "canonicalurl": "https://example.com/event",
            }
        ]
    )


def test_create_documents():
    events = get_sample_events()

    documents = create_documents(events)

    assert len(documents) == 1
    assert "Concert de jazz" in documents[0].page_content
    assert "Entrée gratuite" in documents[0].page_content


def test_document_metadata():
    events = get_sample_events()

    document = create_documents(events)[0]

    assert document.metadata["uid"] == "1"
    assert document.metadata["title"] == "Concert de jazz"
    assert document.metadata["city"] == "Metz"
    assert document.metadata["location"] == "Arsenal"


def test_split_documents():
    events = get_sample_events()

    documents = create_documents(events)
    chunks = split_documents(documents)

    assert len(chunks) >= 1


def test_chunk_size():
    events = get_sample_events()

    documents = create_documents(events)
    chunks = split_documents(documents)

    assert all(
        len(chunk.page_content) <= 1000
        for chunk in chunks
    )


def test_chunk_metadata_are_preserved():
    events = get_sample_events()

    documents = create_documents(events)
    chunks = split_documents(documents)

    assert all(
        chunk.metadata["uid"] == "1"
        for chunk in chunks
    )