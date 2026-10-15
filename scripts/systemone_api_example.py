"""Directly runnable /v1/systemone calling examples: GUI-Actor grounding and clef-flash decisions.

Both go to one address -- the gateway on :8020 with the routes `gui-actor` and `clef-flash` that
scripts/serve_stack.sh starts (a bare kev.serve on its own port works too; pass --base-url):

    uv run python scripts/systemone_api_example.py                        # GUI-Actor: 截图 + 指令 -> 点击点
    uv run python scripts/systemone_api_example.py --example clef-flash   # clef-flash: 状态 + 选项 -> 决策

Each example is the whole request it sends, so the file doubles as the field-mapping reference.

GUI-Actor (kev.gui_actor_model, model id `gui-actor`)
    state        -> the system prompt (kev.api.render; empty falls back to the checkpoint's own prompt)
    instructions -> the user instruction -- one grounding pass per choice question
    images       -> base64 PNG/JPEG (raw or "data:image/png;base64,..."); the LAST one is the screenshot
    criteria     -> required by the API (1..255 entries) and never read by the model
    answer       -> probabilities keyed by *patch index* ("0".."n-1"), not by criteria: `y = idx // n_width`,
                    `x = idx % n_width`, patch centre `((x+0.5)/n_width, (y+0.5)/n_height)`, normalized to
                    the model's own (smart-resized) input, `n_width*patch_pixels` by
                    `n_height*patch_pixels`. `point` is the server's decode of that distribution (the
                    activation-weighted centre of the best connected region) and is what most callers want.
                    `confidence` is the probability mass the patch distribution puts on the predicted
                    region, at the checkpoint's fitted `temperature` (`config.pointer_temperature`, 1.0
                    when unfitted); `confidence_raw` is the same at T=1. The temperature rescales the
                    scalar only -- `point`/`choice` are always decoded at T=1.
                    A request with N choice questions costs N grounding passes; no image, or a noul/score
                    question, is refused with 422.

clef-flash (kev.clef_model, model id `clef-flash`)
    state        -> the screen state the model reads (goal + recentActions)
    questions    -> `instructions` is the question, `criteria` the offered options
    answer       -> a choice over the criteria: `choice` is an option name and `probabilities` is keyed by
                    option name (noul answers carry `noul`, score answers `score` + a legend instead)

    --annotate out.png   GUI-Actor: draw the predicted point on the screenshot and save it
    --raw                print the request (image elided) and the whole response body
"""
import argparse, base64, json, pathlib, sys, time, urllib.error, urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def call(base_url, body, timeout=300):
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


def gui_actor_example(a):
    """Screenshot + instruction -> click point + per-patch probabilities."""
    body = {"model": "gui-actor", "state": a.system,
            "images": [base64.b64encode(pathlib.Path(a.image).read_bytes()).decode()],
            "questions": {a.question: {"type": "choice", "instructions": a.instruction, "criteria": {"1": None}}}}
    if a.raw:
        print("request:", json.dumps({**body, "images": ["<base64 ...>"]}, ensure_ascii=False, indent=2))
    t = time.time()
    resp = call(a.base_url, body, a.timeout)
    print(f"{a.base_url} answered in {time.time() - t:.2f}s (server latency {resp.get('latency_ms')} ms, "
          f"input_tokens {resp['usage']['input_tokens']})")
    if a.raw:
        print("response:", json.dumps(resp, ensure_ascii=False, indent=2))

    ans = resp["answers"][a.question]
    print(f"grid {ans['n_width']}x{ans['n_height']} = {len(ans['probabilities'])} patches "
          f"({ans['patch_pixels']} px each); sum(probabilities) = {sum(ans['probabilities'].values()):.4f}")
    print(f"choice {ans['choice']} (argmax patch, confidence {ans['confidence']} at T={ans['temperature']}, raw {ans['confidence_raw']})")
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


def clef_flash_example(a):
    """Screen state + questions -> a choice over the criteria."""
    body = {"model": "clef-flash",
            "state": {"goal": a.goal, "recentActions": []},
            "questions": {"operation": {"type": "choice",
                                        "instructions": "Choose one operation that advances the goal.",
                                        "criteria": {"TAP": "Tap an observed control to navigate toward the goal.",
                                                     "BACK": "Navigate back one screen.",
                                                     "SCROLL_DOWN": "Scroll down to reveal more content."}}}}
    if a.raw:
        print("request:", json.dumps(body, ensure_ascii=False, indent=2))
    t = time.time()
    resp = call(a.base_url, body, a.timeout)
    print(f"{a.base_url} answered in {time.time() - t:.2f}s (server latency {resp.get('latency_ms')} ms, "
          f"input_tokens {resp['usage']['input_tokens']})")
    if a.raw:
        print("response:", json.dumps(resp, ensure_ascii=False, indent=2))

    ans = resp["answers"]["operation"]
    print(f"choice {ans['choice']} (confidence {ans['confidence']})  probabilities {ans['probabilities']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", choices=["gui-actor", "clef-flash"], default="gui-actor")
    ap.add_argument("--base-url", default="http://127.0.0.1:8020",
                    help="the gateway scripts/serve_stack.sh starts, or a bare kev.serve on its own port")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--raw", action="store_true", help="print the request (image elided) and the whole response body")
    g = ap.add_argument_group("gui-actor")
    g.add_argument("--image", default="docs/alipay.png", help="screenshot to ground on (PNG/JPEG)")
    g.add_argument("--instruction", default="交话费", help="the user instruction, e.g. 'Click on view results'")
    g.add_argument("--system", default="", help="system prompt text; empty uses the checkpoint's own grounding prompt")
    g.add_argument("--question", default="tap_target", help="question id in the request/answers")
    g.add_argument("--top", type=int, default=3, help="how many region points to print (0 = none)")
    g.add_argument("--annotate", metavar="OUT.png", help="draw the predicted point on the image and save it")
    c = ap.add_argument_group("clef-flash")
    c.add_argument("--goal", default="Go to the mobile category after closing the pop up and browse the products",
                   help="the task goal the demo state carries")
    a = ap.parse_args()
    (gui_actor_example if a.example == "gui-actor" else clef_flash_example)(a)


if __name__ == "__main__":
    main()
