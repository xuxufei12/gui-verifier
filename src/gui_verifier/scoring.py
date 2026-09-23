import math

from .schema import Decision, Question


def softmax(values, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Candidate logits must be nonempty and finite")
    scaled = [x / temperature for x in values]
    maximum = max(scaled)
    weights = [math.exp(x - maximum) for x in scaled]
    total = sum(weights)
    return [x / total for x in weights]


def decide(question: Question, logits: list[float], min_probability=0.6,
           min_margin=0.1, temperature=1.0, order_consistent=None):
    if len(logits) != len(question.options):
        raise ValueError("One logit is required per option")
    probs = softmax(logits, temperature)
    ranking = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    top, second = ranking[:2]
    selected = question.options[top].id
    margin = probs[top] - probs[second]
    reasons = []
    if probs[top] < min_probability:
        reasons.append("low_candidate_probability")
    if margin < min_margin:
        reasons.append("small_margin")
    if order_consistent is False:
        reasons.append("option_order_disagreement")
    status = "review" if reasons else "decided"
    if selected == question.unknown_option_id:
        reasons.append("insufficient_visual_evidence")
        status = "unknown"
    return Decision(question_id=question.id, selected_option_id=selected,
                    status=status, probabilities={o.id: p for o, p in zip(question.options, probs)},
                    candidate_logits={o.id: v for o, v in zip(question.options, logits)},
                    max_probability=probs[top], margin=margin, reasons=reasons,
                    order_consistent=order_consistent)
