"""Gradio panel for experimental, non-destructive Applio index compression."""

from pathlib import Path

import gradio as gr

from assets.i18n.i18n import I18nAuto
from rvc.lib.tools.index_optimizer import optimize_file, self_test
from rvc.lib.user_config import get_logs_dir

i18n = I18nAuto()


def _index_choices():
    root = Path(get_logs_dir()).expanduser().resolve()
    if not root.is_dir():
        return []
    return [
        (str(path.relative_to(root)), str(path))
        for path in sorted(root.rglob("*.index"))
        if path.is_file()
        and not path.stem.endswith(("_fp16", "_sq8"))
    ]


def refresh_choices():
    return gr.update(choices=_index_choices(), value=[])


def optimize_selected(selected, optimize_all, mode):
    available = {value for _, value in _index_choices()}
    selected = selected or []
    files = sorted(available if optimize_all else set(selected))
    if not files:
        return "No indexes selected. Generate an index for a trained voice first."
    if not set(files).issubset(available):
        return "Selection is stale. Refresh the index list first."
    methods = ("fp16", "sq8") if mode == "both" else (mode,)
    lines = []
    for filename in files:
        source = Path(filename)
        for method in methods:
            destination = source.with_name(f"{source.stem}_{method}.index")
            try:
                result = optimize_file(
                    source, destination, method,
                    chunk_size=4096, train_rows=10000, check_rows=64
                )
                if result:
                    lines.append(
                        f"OK {source.name} ({method}): "
                        f"{result['saving_percent']:.1f}% smaller; "
                        f"top-8 overlap "
                        f"{result['diagnostics']['mean_top8_id_overlap']:.1%}"
                    )
                    lines.append(f"Saved {destination}")
                else:
                    lines.append(
                        f"SKIP {source.name} ({method}): already exists "
                        "or compression did not reduce file size"
                    )
            except Exception as exc:
                lines.append(f"ERROR {source.name} ({method}): {exc}")
    lines.append(
        "Original .index and .pth files are unchanged. "
        "Compare converted audio before removing any original."
    )
    return "\n".join(lines)


def run_compatibility_test():
    try:
        self_test()
        return (
            "PASS: synthetic Flat and IVF FP16/SQ8 round-trips succeeded. "
            "This does not establish identical converted audio quality."
        )
    except Exception as exc:
        return f"FAIL: {type(exc).__name__}: {exc}"


def index_optimizer_tab():
    gr.Markdown(i18n(
        "Compress existing FAISS `.index` files using FP16 or SQ8 without "
        "changing voice `.pth` files or overwriting originals. "
        "**Experimental:** A/B test audio quality before removing any files."
    ))
    sources = gr.Dropdown(
        label=i18n("Indexes to optimize"),
        info=i18n("Select original indexes from your Applio logs folder."),
        choices=_index_choices(),
        value=[],
        multiselect=True,
        interactive=True,
    )
    with gr.Row():
        refresh = gr.Button(i18n("Refresh indexes"))
        all_indexes = gr.Checkbox(
            label=i18n("Optimize all original indexes"), value=False
        )
    method = gr.Dropdown(
        label=i18n("Compression format"),
        choices=[
            (i18n("FP16 (recommended first)"), "fp16"),
            (i18n("SQ8 (smaller, more experimental)"), "sq8"),
            (i18n("Both (compare sizes and quality)"), "both"),
        ],
        value="fp16",
        interactive=True,
    )
    with gr.Row():
        test_button = gr.Button(i18n("Run compatibility self-test"))
        optimize_button = gr.Button(
            i18n("Optimize indexes"), variant="primary"
        )
    summary = gr.Textbox(
        label=i18n("Optimization report"),
        value="",
        lines=7,
        max_lines=18,
        interactive=False,
    )
    refresh.click(fn=refresh_choices, outputs=[sources])
    test_button.click(fn=run_compatibility_test, outputs=[summary])
    optimize_button.click(
        fn=optimize_selected,
        inputs=[sources, all_indexes, method],
        outputs=[summary],
    )
