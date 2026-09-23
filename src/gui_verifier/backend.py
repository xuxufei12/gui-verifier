import time
from threading import Lock

from .prompt import label_token_ids, messages_for
from .schema import VerifyRequest, VerifyResult
from .scoring import decide

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"


class QwenVerifier:
    """Independent question batches, one forward per batch, no autoregressive decoding.

    v0.1 repeats the visual encoder per question; it does NOT claim shared KV/vision caching.
    """

    def __init__(self, model=DEFAULT_MODEL, device="auto", batch_size=1,
                 max_pixels=512 * 512, temperature=1.0, adapter=None, revision=None):
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        if batch_size < 1 or max_pixels < 4096:
            raise ValueError("batch_size >= 1 and max_pixels >= 4096 required")
        self.torch = torch
        self.model_name = model
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(model, revision=revision)
        self.processor.tokenizer.padding_side = "left"
        # Bound both axes through the processor's area budget; keep aspect ratio intact.
        self.processor.image_processor.size = {"shortest_edge": 64 * 64, "longest_edge": max_pixels}
        self.processor.image_processor.min_pixels = 64 * 64
        self.processor.image_processor.max_pixels = max_pixels
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model, revision=revision, dtype=dtype, attn_implementation="sdpa"
        ).to(self.device)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.label_ids = label_token_ids(self.processor.tokenizer)
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_pixels = max_pixels
        self.adapter = adapter
        self.lock = Lock()
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def prepare(self, image, request, questions):
        conversations = [messages_for(image, request, q) for q in questions]
        inputs = self.processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=True, return_tensors="pt", padding=True,
        )
        if inputs["input_ids"].shape[1] > 8192:
            raise ValueError("Prompt exceeds 8192 tokens; shorten rules or reduce image resolution")
        return inputs.to(self.device)

    def _sync(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def _score(self, image, request, questions):
        logits, preprocess, forward = [], 0.0, 0.0
        for start in range(0, len(questions), self.batch_size):
            chunk = questions[start:start + self.batch_size]
            begin = time.perf_counter()
            inputs = self.prepare(image, request, chunk)
            self._sync()
            middle = time.perf_counter()
            with self.torch.inference_mode():
                output = self.model(**inputs, use_cache=False, logits_to_keep=1)
                scores = output.logits[:, -1, self.label_ids].float().cpu().tolist()
            self._sync()
            end = time.perf_counter()
            logits.extend([row[:len(q.options)] for row, q in zip(scores, chunk)])
            preprocess += middle - begin
            forward += end - middle
        return logits, preprocess, forward

    def verify(self, image, request: VerifyRequest):
        started = time.perf_counter()
        with self.lock:
            logits, prep, forward = self._score(image, request, request.questions)
            consistency = [None] * len(logits)
            if request.check_order:
                reversed_qs = [q.model_copy(update={"options": list(reversed(q.options))})
                               for q in request.questions]
                other, p2, f2 = self._score(image, request, reversed_qs)
                prep += p2
                forward += f2
                for i, (original, reversed_scores) in enumerate(zip(logits, other)):
                    a = max(range(len(original)), key=original.__getitem__)
                    b = max(range(len(reversed_scores)), key=reversed_scores.__getitem__)
                    consistency[i] = a == len(original) - 1 - b
            decisions = [decide(q, scores, request.min_probability, request.min_margin,
                                self.temperature, stable)
                         for q, scores, stable in zip(request.questions, logits, consistency)]
        return VerifyResult(
            model=self.model_name, decisions=decisions,
            timing_ms={"preprocess": prep * 1000, "forward": forward * 1000,
                       "total": (time.perf_counter() - started) * 1000},
            metadata={"device": self.device, "batch_size": self.batch_size,
                      "temperature": self.temperature, "calibrated": False,
                      "score_semantics": "softmax restricted to provided option tokens; not correctness probability",
                      "shared_visual_encoding": False, "shared_prefix_cache": False,
                      "generated_tokens": 0, "adapter": self.adapter,
                      "model_revision": self.revision, "max_pixels": self.max_pixels,
                      "image_size": list(image.size), "order_check": request.check_order},
        )
