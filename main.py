from pathlib import Path
import gradio as gr
from rag import ingest, ask, kb_info, KB_FOLDER

SAMPLE_QUERIES = [
    "Is it possible to convince a recruit to accept 10% less static pay? Show me how.",
    "How do I handle a candidate with a competing offer 15% higher than ours?",
    "What non-monetary benefits should I highlight when cash budget is tight?",
    "How do I reduce someone's salary during restructuring without losing them?",
    "When does a joining bonus work and when does it NOT work?",
    "The candidate is anchoring very high. How do I reframe the negotiation?",
    "My recruit has high EMIs. Can I still close a salary gap of ₹2L?",
]

CSS = """
footer { display: none !important; }
.gradio-container { max-width: 1300px !important; margin: auto; }
"""


def upload_files(files):
    if not files:
        return "No files selected."
    msgs = []
    for f in files:
        src  = Path(f.name)
        dest = KB_FOLDER / src.name
        dest.write_bytes(src.read_bytes())
        msgs.append(f"✅ {src.name}")
    return "\n".join(msgs) + "\n\nClick Ingest KB to process."


def do_ingest(clear):
    return ingest(clear_first=clear)


def do_ask(query, top_k, use_expansion, history):
    if not query.strip():
        return history, history, "", "", ""

    history = history or []

    try:
        answer, chunks, queries = ask(query, int(top_k), use_expansion)
    except Exception as e:
        answer  = f"❌ Error: {e}\n\nMake sure `ollama serve` is running."
        chunks  = []
        queries = [query]

    history.append({"role": "user",      "content": query})
    history.append({"role": "assistant", "content": answer})

    sources_md = "### Retrieved Sources\n"
    for i, c in enumerate(chunks, 1):
        bar = "█" * int(c["score"] * 10) + "░" * (10 - int(c["score"] * 10))
        sources_md += f"**{i}. {c['source']}** `{c['score']}` `{bar}`\n"
        sources_md += f"> {c['text'][:200].strip()}...\n\n"

    expanded_md = "**Queries used:**\n" + "\n".join(f"- `{q}`" for q in queries)

    return history, history, sources_md, expanded_md, ""


with gr.Blocks(title="HR Negotiation RAG") as demo:

    gr.HTML("""
    <div style="text-align:center; padding:18px 0 10px">
      <h1 style="font-size:2rem; font-weight:800; color:#e8e8e8; letter-spacing:-0.5px">
        🤝 HR Salary Negotiation RAG
      </h1>
    </div>
    """)

    with gr.Tabs():

        with gr.TabItem("💬 Chat"):
            chat_state = gr.State([])

            with gr.Row(equal_height=False):

                with gr.Column(scale=1, min_width=260):
                    gr.Markdown("#### ⚙️ Settings")
                    top_k_slider  = gr.Slider(1, 8, value=4, step=1, label="Top-K chunks")
                    use_expansion = gr.Checkbox(value=True, label="Query Expansion")

                    gr.Markdown("#### 💡 Sample Queries")
                    sample_btns = []
                    for sq in SAMPLE_QUERIES:
                        b = gr.Button(
                            sq[:68] + ("…" if len(sq) > 68 else ""),
                            size="sm", variant="secondary"
                        )
                        sample_btns.append((b, sq))

                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="HR Assistant",
                        height=460,
                    )
                    with gr.Row():
                        query_box = gr.Textbox(
                            placeholder="Ask a salary negotiation question...",
                            lines=2,
                            scale=5,
                            show_label=False,
                        )
                        with gr.Column(scale=1, min_width=90):
                            ask_btn   = gr.Button("Ask ➤",  variant="primary")
                            clear_btn = gr.Button("🗑 Clear", size="sm")

            with gr.Accordion("📚 Sources & Query Details", open=False):
                with gr.Row():
                    sources_box  = gr.Markdown()
                    expanded_box = gr.Markdown()

            for btn, sq in sample_btns:
                btn.click(fn=lambda q=sq: q, outputs=query_box)

            ask_btn.click(
                fn=do_ask,
                inputs=[query_box, top_k_slider, use_expansion, chat_state],
                outputs=[chat_state, chatbot, sources_box, expanded_box, query_box],
            )
            query_box.submit(
                fn=do_ask,
                inputs=[query_box, top_k_slider, use_expansion, chat_state],
                outputs=[chat_state, chatbot, sources_box, expanded_box, query_box],
            )
            clear_btn.click(
                fn=lambda: ([], [], "", ""),
                outputs=[chat_state, chatbot, sources_box, expanded_box],
            )

        with gr.TabItem("📂 Knowledge Base"):
            gr.Markdown("Drop `.txt` `.pdf` `.docx` `.json` files below, then click **Ingest KB**.")

            with gr.Row():
                with gr.Column():
                    upload_box    = gr.File(
                        label="Upload to KB", file_count="multiple",
                        file_types=[".txt", ".md", ".pdf", ".docx", ".json", ".csv"]
                    )
                    upload_btn    = gr.Button("📤 Upload", variant="secondary")
                    upload_status = gr.Textbox(label="Upload Status", lines=3, interactive=False)

                with gr.Column():
                    clear_chk  = gr.Checkbox(label="Clear vectors before ingesting", value=False)
                    ingest_btn = gr.Button("⚡ Ingest KB", variant="primary")
                    ingest_log = gr.Textbox(label="Ingest Log", lines=10, interactive=False)

            refresh_btn   = gr.Button("🔄 Refresh Status")
            kb_status_box = gr.Markdown()

            upload_btn.click(fn=upload_files, inputs=upload_box, outputs=upload_status)
            ingest_btn.click(fn=do_ingest,    inputs=clear_chk,  outputs=ingest_log)
            refresh_btn.click(fn=kb_info,                        outputs=kb_status_box)
            demo.load(fn=kb_info,                                outputs=kb_status_box)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True, css=CSS)