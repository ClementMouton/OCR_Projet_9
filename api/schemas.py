from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    uid: str | None = None
    title: str | None = None
    url: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


class RebuildResponse(BaseModel):
    message: str
    events_count: int
    chunks_count: int
    vectors_count: int