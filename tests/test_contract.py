import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from gui_verifier.api import create_app
from gui_verifier.prompt import messages_for
from gui_verifier.schema import VerifyRequest, VerifyResult
from gui_verifier.scoring import decide, softmax


def task():
    return VerifyRequest.model_validate({"goal":"Log in", "questions":[
        {"id":"q1", "text":"Is login complete?", "unknown_option_id":"unknown", "options":[
            {"id":"yes","text":"Yes"},{"id":"no","text":"No"},{"id":"unknown","text":"Unknown"}]}]})


def test_review_is_not_a_fabricated_unknown_prediction():
    d = decide(task().questions[0], [1.0, 0.99, -1])
    assert d.selected_option_id == "yes" and d.status == "review"
    assert "small_margin" in d.reasons
    assert sum(d.probabilities.values()) == pytest.approx(1)


def test_explicit_unknown_and_order_instability():
    q = task().questions[0]
    assert decide(q, [-4, -4, 4]).status == "unknown"
    assert decide(q, [4, -4, -4], order_consistent=False).status == "review"


def test_unique_ids_and_unknown_contract():
    raw = task().model_dump()
    raw["questions"][0]["unknown_option_id"] = "missing"
    with pytest.raises(ValidationError):
        VerifyRequest.model_validate(raw)
    raw = task().model_dump()
    raw["questions"] *= 2
    with pytest.raises(ValidationError):
        VerifyRequest.model_validate(raw)


def test_question_isolation():
    r = task()
    other = r.questions[0].model_copy(update={"id":"q2", "text":"SECRET_OTHER_QUESTION"})
    r.questions.append(other)
    text = str(messages_for("image-placeholder", r, r.questions[0]))
    assert "SECRET_OTHER_QUESTION" not in text


def test_numeric_stability():
    assert softmax([10000, 10000]) == [0.5, 0.5]
    with pytest.raises(ValueError):
        softmax([float("nan"), 0])
    with pytest.raises(ValueError):
        softmax([1, 2], 0)


class StubVerifier:
    def verify(self, image, request):
        assert image.mode == "RGB"
        return VerifyResult(model="test-stub", decisions=[decide(request.questions[0], [4, 0, -1])],
                            timing_ms={}, metadata={"test_stub": True})


def test_api_roundtrip_and_reject_bad_image():
    client = TestClient(create_app(StubVerifier()))
    buf = io.BytesIO()
    Image.new("RGB", (32,32)).save(buf, format="PNG")
    response = client.post("/v1/verify", files={"image":("test.png",buf.getvalue(),"image/png")},
                           data={"request":task().model_dump_json()})
    assert response.status_code == 200
    assert response.json()["decisions"][0]["selected_option_id"] == "yes"
    assert response.json()["timing_ms"]["server_total"] >= 0
    bad = client.post("/v1/verify", files={"image":("x.png",b"not an image","image/png")},
                      data={"request":task().model_dump_json()})
    assert bad.status_code == 422


def test_demo_page_and_unloaded_health():
    client = TestClient(create_app())
    assert client.get("/health").json()["model_loaded"] is False
    assert "GUI Verifier" in client.get("/").text
