from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Option(StrictModel):
    id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=1000)


class Question(StrictModel):
    id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=2000)
    options: list[Option] = Field(min_length=2, max_length=12)
    unknown_option_id: str

    @model_validator(mode="after")
    def check_options(self):
        ids = [o.id for o in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError("Option IDs must be unique within each question")
        if self.unknown_option_id not in ids:
            raise ValueError("unknown_option_id must identify an explicit insufficient-evidence option")
        return self


class VerifyRequest(StrictModel):
    goal: str = Field(min_length=1, max_length=4000)
    rules: list[str] = Field(default_factory=list, max_length=32)
    questions: list[Question] = Field(min_length=1, max_length=32)
    min_probability: float = Field(default=0.6, ge=0, le=1)
    min_margin: float = Field(default=0.1, ge=0, le=1)
    check_order: bool = False

    @model_validator(mode="after")
    def check_request(self):
        ids = [q.id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("Question IDs must be unique")
        if any(not r.strip() or len(r) > 2000 for r in self.rules):
            raise ValueError("Rules must contain 1..2000 characters")
        return self


class Decision(StrictModel):
    question_id: str
    selected_option_id: str
    status: Literal["decided", "unknown", "review"]
    probabilities: dict[str, float]
    candidate_logits: dict[str, float]
    max_probability: float
    margin: float
    reasons: list[str]
    order_consistent: bool | None = None


class VerifyResult(StrictModel):
    model: str
    decisions: list[Decision]
    timing_ms: dict[str, float]
    metadata: dict
