"""Call a GUI-Actor /v1/systemone server: screenshot + instruction -> click point + per-patch probabilities.

The server is kev.serve.GUIActorServer (`python -m kev.serve --run microsoft/GUI-Actor-<...>`); this
script is the client, and doubles as the field-mapping reference:

    state        -> the system prompt (kev.api.render; empty falls back to the checkpoints' own prompt)
    instructions -> the user instruction -- one grounding pass per choice question
    images       -> base64 PNG/JPEG (raw or "data:image/png;base64,..."); the LAST one is the screenshot
    criteria     -> required by the API (1..255 entries) and never read by the model

The answer's probabilities are keyed by *patch index* ("0".."n-1"), not by criteria: `y = idx // n_width`,
`x = idx % n_width`, and a patch centre is `((x+0.5)/n_width, (y+0.5)/n_height)` -- normalized to the
model's own (smart-resized) input, `n_width*patch_pixels` by `n_height*patch_pixels`. `point` is the
server's decode of that distribution (the activation-weighted centre of the best connected region) and
is what most callers want. A request with N choice questions costs N grounding passes, so send only the
question you need grounded; a request with no image, or with a noul/score question, is refused with 422.

Usage:
    uv run python scripts/gui_actor_api.py \
        --image ../GUI-Actor/android_control_parsered/parsered/13628/step_006_screenshot.png \
        --instruction "Click on view results"

    --annotate out.png   draw the predicted point on the screenshot and save it
    --raw                print the whole response body (the per-patch distribution included)
    --top 3              list the top regions (point, mean activation)
"""
import argparse, base64, json, pathlib, sys, time, urllib.error, urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def call(base_url, body, timeout):
    """POST /v1/systemone; an HTTP error's detail is the server's message (a 422 explains the refusal)."""
    req = urllib.request.Request(f"{base_url.rstrip('/')}/v1/systemone", data=json.dumps(body).encode(),
                                 method="POST", headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try: detail = json.loads(detail).get("detail", detail)
        except ValueError: pass
        raise SystemExit(f"HTTP {e.code}: {detail}")


def annotate(path, image_path, point):
    """Draw the predicted point (normalized) on the screenshot and save it to `path`."""
    from PIL import Image, ImageDraw
    image = Image.open(image_path).convert("RGB")
    x, y = point[0] * image.width, point[1] * image.height
    r = max(8, image.width // 80)
    d = ImageDraw.Draw(image)
    d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=max(2, r // 4))
    d.line([x - 2 * r, y, x + 2 * r, y], fill=(255, 0, 0), width=max(2, r // 4))
    d.line([x, y - 2 * r, x, y + 2 * r], fill=(255, 0, 0), width=max(2, r // 4))
    image.save(path)
    print(f"annotated point ({x:.0f}, {y:.0f}) px written to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="docs/alipay.png", help="screenshot to ground on (PNG/JPEG)")
    ap.add_argument("--instruction", default="交话费", help="the user instruction, e.g. 'Click on view results'")
    ap.add_argument("--state", default="", help="system prompt text; empty uses the checkpoints' own grounding prompt")
    ap.add_argument("--question", default="tap_target", help="question id in the request/answers")
    ap.add_argument("--base-url", default="http://127.0.0.1:8010")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--top", type=int, default=3, help="how many region points to print (0 = none)")
    ap.add_argument("--annotate", metavar="OUT.png", help="draw the predicted point on the image and save it")
    ap.add_argument("--raw", action="store_true", help="print the whole response body")
    a = ap.parse_args()

    image_b64 = base64.b64encode(pathlib.Path(a.image).read_bytes()).decode()
    body = {"model": a.model, "state": a.state, "images": [image_b64],
            "questions": {a.question: {"type": "choice", "instructions": a.instruction, "criteria": {"1": None}}}}
    t = time.time()
    resp = call(a.base_url, body, a.timeout)
    print(f"{a.base_url} answered in {time.time() - t:.2f}s "
          f"(server latency {resp.get('latency_ms')} ms, input_tokens {resp['usage']['input_tokens']})")
    if a.raw:
        print(json.dumps(resp, indent=2))

    ans = resp["answers"][a.question]
    n_patches = len(ans["probabilities"])
    print(f"grid {ans['n_width']}x{ans['n_height']} = {n_patches} patches ({ans['patch_pixels']} px each)")
    print(f"choice {ans['choice']} (confidence {ans['confidence']}); "
          f"sum(probabilities) = {sum(ans['probabilities'].values()):.4f}")
    x, y = ans["point"]
    W, H = ans["n_width"] * ans["patch_pixels"], ans["n_height"] * ans["patch_pixels"]
    from PIL import Image
    ow, oh = Image.open(a.image).size
    print(f"point {x:.4f}, {y:.4f}  =  ({x * W:.0f}, {y * H:.0f}) px on the model's {W}x{H} input"
          f"  ~  ({x * ow:.0f}, {y * oh:.0f}) px of the original {ow}x{oh}")
    for i, (p, v) in enumerate(zip(ans["topk_points"][:a.top], ans["topk_values"][:a.top])):
        print(f"  region {i}: ({p[0]:.4f}, {p[1]:.4f})  mean activation {v:.4f}")
    if a.annotate:
        annotate(a.annotate, a.image, ans["point"])


if __name__ == "__main__":
    main()
