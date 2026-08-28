from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
)


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class AskRequest(StrictBaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Question en français concernant les événements disponibles à Metz.",
        examples=["Quels concerts sont disponibles à Metz ?"],
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not any(char.isalpha() for char in value):
            raise ValueError(
                "La question doit contenir au moins un caractère alphabétique."
            )

        return value


class Source(StrictBaseModel):
    uid: str | None = None
    title: str | None = None
    url: HttpUrl | None = None
    start_date: str | None = None
    end_date: str | None = None
    location: str | None = None


class AskResponse(StrictBaseModel):
    question: str
    answer: str
    sources: list[Source]


class RebuildResponse(StrictBaseModel):
    message: str
    events_count: int = Field(ge=0)
    chunks_count: int = Field(ge=0)
    vectors_count: int = Field(ge=0)