import argparse, os, re, json, random, sys, time, threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import gradio as gr
from PIL import Image  # added for right-half cropping

IDX_RE = re.compile(r"val_(\d+)_result\.(png|jpg|jpeg)$", re.IGNORECASE)

def _index_from_name(p: Path) -> Optional[int]:
    m = IDX_RE.search(p.name)
    if m:
        return int(m.group(1))
    # fallback: first integer in filename
    nums = re.findall(r"\d+", p.stem)
    return int(nums[0]) if nums else None

def _collect_pairs(dir_a: Path, dir_b: Path) -> List[int]:
    a_map, b_map = {}, {}
    for p in dir_a.glob("*.png"):
        i = _index_from_name(p);  a_map[i] = p if i is not None else None
    for p in dir_a.glob("*.jpg"):
        i = _index_from_name(p);  a_map[i] = p if i is not None else None
    for p in dir_b.glob("*.png"):
        i = _index_from_name(p);  b_map[i] = p if i is not None else None
    for p in dir_b.glob("*.jpg"):
        i = _index_from_name(p);  b_map[i] = p if i is not None else None
    idxs = sorted(set(k for k in a_map.keys() if k is not None)
                  & set(k for k in b_map.keys() if k is not None))
    return idxs

def _result_path(root: Path, idx: int) -> Optional[Path]:
    # canonical names first
    cand = root / f"val_{idx:05d}_result.png"
    if cand.exists(): return cand
    cand = root / f"val_{idx:05d}_result.jpg"
    if cand.exists(): return cand
    # fallback: scan
    for p in root.glob(f"*{idx:05d}*result*.png"):
        return p
    for p in root.glob(f"*{idx:05d}*result*.jpg"):
        return p
    return None

def _source_path(root: Path, idx: int) -> Optional[Path]:
    for ext in ("png", "jpg", "jpeg"):
        cand = root / f"val_{idx:05d}_source.{ext}"
        if cand.exists(): return cand
    return None

def _mask_path(root: Path, idx: int) -> Optional[Path]:
    for ext in ("png", "jpg", "jpeg"):
        cand = root / f"val_{idx:05d}_mask.{ext}"
        if cand.exists(): return cand
    return None

def build_order(dir_a: Path, dir_b: Path, shuffle: bool) -> List[int]:
    idxs = _collect_pairs(dir_a, dir_b)
    if shuffle:
        random.Random(1337).shuffle(idxs)
    return idxs

def stats_md(votes: Dict[int, str], total_pairs: int) -> str:
    # Only count rated (exclude "skip")
    rated = {k: v for k, v in votes.items() if v in ("A", "B", "T")}
    better = sum(1 for v in rated.values() if v == "B")  # B=next level better
    worse  = sum(1 for v in rated.values() if v == "A")
    tie    = sum(1 for v in rated.values() if v == "T")
    skipped= sum(1 for v in votes.values() if v == "S")

    denom = max(1, len(rated))
    promo_score = (better + tie) / denom
    return (
        f"**Progress:** {len(rated)}/{total_pairs} rated  |  **Skipped:** {skipped}\n\n"
        f"**A better:** {worse}   |   **Tie:** {tie}   |   **B better:** {better}\n\n"
        f"**Promotion score** = (B + Tie) / Rated = **{promo_score:.1%}**"
    )

def save_results(out_path: Path,
                 name_a: str, name_b: str,
                 dir_a: Path, dir_b: Path,
                 idx_order: List[int],
                 votes: Dict[int, str]) -> Dict:
    rated = {k: v for k, v in votes.items() if v in ("A", "B", "T")}
    better = sum(1 for v in rated.values() if v == "B")
    worse  = sum(1 for v in rated.values() if v == "A")
    tie    = sum(1 for v in rated.values() if v == "T")
    data = {
        "level_a": name_a,
        "level_b": name_b,
        "val_dir_a": str(dir_a),
        "val_dir_b": str(dir_b),
        "order": idx_order,
        "votes": {str(k): v for k, v in votes.items()},  # raw per-index votes
        "total_comparisons": len(rated),                 # only rated (no skips)
        "better_count": better,
        "worse_count": worse,
        "neutral_count": tie,
        "skipped_count": sum(1 for v in votes.values() if v == "S"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return data

def make_app(args):
    dir_a = Path(args.dir_a).resolve()
    dir_b = Path(args.dir_b).resolve()
    if not dir_a.exists() or not dir_b.exists():
        print("Validation directories not found.", file=sys.stderr)
        sys.exit(2)

    idx_order = build_order(dir_a, dir_b, shuffle=not args.no_shuffle)
    total_pairs = len(idx_order)
    if total_pairs == 0:
        print("No paired validation images found.", file=sys.stderr)
        sys.exit(3)

    start_pos = 0
    if args.start_index is not None and args.start_index in idx_order:
        start_pos = idx_order.index(args.start_index)

    with gr.Blocks(title=f"Compare: {args.name_a} vs {args.name_b}") as demo:
        gr.Markdown(f"### Human evaluation: **{args.name_a} (A)** vs **{args.name_b} (B)**")
        gr.Markdown("Choose which result looks better. “Tie” if about the same. You can skip and come back later.")

        state_idx_order = gr.State(idx_order)   # List[int]
        state_pos       = gr.State(start_pos)   # int pointer into idx_order
        state_votes     = gr.State({})          # Dict[int, str]

        with gr.Row():
            img_a = gr.Image(label=f"A — {args.name_a}", interactive=False, height=args.image_height, show_fullscreen_button=True)
            img_b = gr.Image(label=f"B — {args.name_b}", interactive=False, height=args.image_height, show_fullscreen_button=True)

        with gr.Accordion("Show source/mask (from A)", open=False):
            with gr.Row():
                src = gr.Image(label="Source", interactive=False, height=200)
                msk = gr.Image(label="Mask", interactive=False, height=200)

        stat = gr.Markdown(stats_md({}, total_pairs))

        with gr.Row():
            btn_back  = gr.Button("⬅️ Back")
            btn_a     = gr.Button("A is better")
            btn_tie   = gr.Button("Tie / About same")
            btn_b     = gr.Button("B is better")
            btn_skip  = gr.Button("Skip ➡️")
        with gr.Row():
            btn_finish = gr.Button("✅ Finish & Save", variant="primary")
            save_path  = gr.Textbox(value=str(Path(args.out).resolve()), label="Results will be saved to", interactive=False)

        def _right_half(p):
            if p is None:
                return None
            try:
                im = Image.open(p)
                try:
                    w, h = im.size
                    cropped = im.crop((w // 2, 0, w, h)).copy()
                    return cropped
                finally:
                    try:
                        im.close()
                    except Exception:
                        pass
            except Exception:
                # Fallback: if anything goes wrong, return the original path
                return str(p)

        def _load(idx_order, pos):
            idx = idx_order[pos]
            pa = _result_path(dir_a, idx)
            pb = _result_path(dir_b, idx)
            sa = _source_path(dir_a, idx)
            ma = _mask_path(dir_a, idx)
            return (_right_half(pa),
                    _right_half(pb),
                    str(sa) if sa else None,
                    str(ma) if ma else None)

        def _go(idx_order, pos, votes, move):
            # move ∈ {"A", "B", "T", "S", "BACK", "INIT"}
            if move in ("A", "B", "T", "S"):
                idx = idx_order[pos]
                votes[idx] = move
                pos = min(len(idx_order)-1, pos + 1)
            elif move == "BACK":
                pos = max(0, pos - 1)
            else:
                pass

            pa, pb, sa, ma = _load(idx_order, pos)
            md = stats_md(votes, len(idx_order))
            return pa, pb, sa, ma, pos, votes, md

        # wire up events
        for button, code in ((btn_a,"A"), (btn_b,"B"), (btn_tie,"T"), (btn_skip,"S")):
            button.click(
                _go,
                inputs=[state_idx_order, state_pos, state_votes, gr.State(code)],
                outputs=[img_a, img_b, src, msk, state_pos, state_votes, stat],
            )

        btn_back.click(
            _go,
            inputs=[state_idx_order, state_pos, state_votes, gr.State("BACK")],
            outputs=[img_a, img_b, src, msk, state_pos, state_votes, stat],
        )

        def _save_and_exit(idx_order, votes):
            data = save_results(Path(args.out), args.name_a, args.name_b, dir_a, dir_b, idx_order, votes)
            # try to close server shortly after save
            def _shutdown():
                try:
                    # gradio>=4
                    import gradio as _gr
                    _gr.close_all()
                except Exception:
                    pass
                # ensure exit if close_all isn't available
                os._exit(0)
            threading.Timer(1.0, _shutdown).start()
            return f"Saved. better={data['better_count']}, tie={data['neutral_count']}, worse={data['worse_count']}. Exiting..."

        btn_finish.click(
            _save_and_exit,
            inputs=[state_idx_order, state_votes],
            outputs=[stat],
        )

        # initial load
        demo.load(
            _go,
            inputs=[state_idx_order, state_pos, state_votes, gr.State("INIT")],
            outputs=[img_a, img_b, src, msk, state_pos, state_votes, stat],
        )

    return demo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_a", type=str, help="Previous/baseline level val/ directory")
    parser.add_argument("dir_b", type=str, help="Current level val/ directory")
    parser.add_argument("name_a", type=str, help="Label for A (prev level)")
    parser.add_argument("name_b", type=str, help="Label for B (current level)")
    parser.add_argument("--out", type=str, default="curriculum_outputs/validation_results.json")
    parser.add_argument("--no-shuffle", action="store_true", help="Keep natural index order")
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_PORT", "7861")))
    parser.add_argument("--server-name", type=str, default=os.environ.get("GRADIO_SERVER","0.0.0.0"))
    parser.add_argument("--share", action="store_true", help="Use Gradio public tunnel")
    parser.add_argument("--auth", type=str, default=os.environ.get("GRADIO_AUTH", ""), help="Format: user:pass")
    args = parser.parse_args()

    app = make_app(args)

    auth_tuple = None
    if args.auth and ":" in args.auth:
        u, p = args.auth.split(":", 1)
        auth_tuple = (u, p)

    # Launch remote-friendly
    app.queue().launch(
        server_name=args.server_name,
        server_port=args.port,
        share=True,
        auth=auth_tuple,
        inbrowser=False,
        show_error=True,
    )

if __name__ == "__main__":
    main()