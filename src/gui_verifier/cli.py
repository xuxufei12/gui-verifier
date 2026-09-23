import argparse
import json
from pathlib import Path

from .schema import VerifyRequest


def main():
    parser = argparse.ArgumentParser(description="Screenshot to independent typed decisions")
    parser.add_argument("--image", required=True)
    parser.add_argument("--request", required=True, help="JSON request; see examples/request.json")
    parser.add_argument("--output")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision")
    parser.add_argument("--adapter")
    parser.add_argument("--fast", action="store_true", help="Merge adapter and share vision encoding; micro-batch <= 4")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    args = parser.parse_args()
    from .api import load_image
    from .backend import QwenVerifier
    if args.fast:
        from .fast_backend import FastQwenVerifier
        QwenVerifier = FastQwenVerifier
    request = VerifyRequest.model_validate_json(Path(args.request).read_text(encoding="utf-8"))
    image = load_image(Path(args.image).read_bytes())
    verifier = QwenVerifier(model=args.model, revision=args.revision, device=args.device,
                            batch_size=args.batch_size, max_pixels=args.max_pixels, adapter=args.adapter)
    result = verifier.verify(image, request).model_dump_json(indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(result, encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
