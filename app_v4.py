import os

import gradio as gr
import torch

from components.tokenizer import decode, encode, tokenizer
from train_script_v4 import TrainConfig, build_model


CFG = TrainConfig()


def get_eot_token_id() -> int:
    eot_tokens = encode("[ENDOFTEXT]")
    if len(eot_tokens) == 1:
        return eot_tokens[0]
    return tokenizer._special_tokens["[ENDOFTEXT]"]


MODEL = build_model(CFG).to(CFG.device)
CKPT_CANDIDATES = [
    os.path.join(CFG.ckpt_dir, CFG.sft2_ckpt_name),
    os.path.join(CFG.ckpt_dir, CFG.sft1_ckpt_name),
    os.path.join(CFG.ckpt_dir, CFG.pretrain_ckpt_name),
]
CKPT_PATH = next((path for path in CKPT_CANDIDATES if os.path.exists(path)), None)
if CKPT_PATH is None:
    raise FileNotFoundError(f"Checkpoint not found. Tried: {CKPT_CANDIDATES}")

state_dict = torch.load(CKPT_PATH, map_location=CFG.device)
MODEL.load_state_dict(state_dict)
MODEL.eval()
EOT_TOKEN_ID = get_eot_token_id()
CONTROL_TOKEN_IDS = {
    tid for tok in ("[STARTOFTEXT]", "[INST]", "[/INST]") for tid in encode(tok)
}
print(f"Loaded checkpoint: {CKPT_PATH}")


def maybe_format_prompt(prompt: str) -> str:
    stripped = prompt.strip()
    if not stripped:
        return " [STARTOFTEXT] [INST] Say hello. [/INST] "
    if "[STARTOFTEXT]" in stripped:
        return stripped
    return f" [STARTOFTEXT] [INST] {stripped} [/INST] "


@torch.no_grad()
def generate_until_eot(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    prompt_text = maybe_format_prompt(prompt)
    token_list = encode(prompt_text)
    prompt_len = len(token_list)
    tokens = torch.tensor(token_list, dtype=torch.long, device=CFG.device).unsqueeze(0)

    min_new_tokens = 16
    visible_tokens: list[int] = []
    for step in range(max_new_tokens):
        ctx = tokens[:, -CFG.block_size :]
        logits = MODEL(ctx)[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        token_id = int(next_token.item())
        tokens = torch.cat([tokens, next_token], dim=1)
        if token_id not in CONTROL_TOKEN_IDS and token_id != EOT_TOKEN_ID:
            visible_tokens.append(token_id)
        if token_id == EOT_TOKEN_ID and step + 1 >= min_new_tokens:
            break

    generated_tokens = tokens[0].tolist()[prompt_len:]
    if visible_tokens:
        return decode(visible_tokens).strip()
    raw_generated = decode(generated_tokens)
    raw_generated = raw_generated.replace("[ENDOFTEXT]", "").strip()
    if raw_generated:
        return raw_generated
    return "(no visible text generated; try temperature 1.0 and top-k 100)"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Thonk v4 Generator") as demo:
        gr.Markdown("## Thonk v4 Text Generator")
        prompt = gr.Textbox(
            label="Prompt",
            lines=6,
            placeholder="Ask a question or give an instruction...",
        )
        with gr.Row():
            max_new_tokens = gr.Slider(1, 512, value=180, step=1, label="Max New Tokens")
            temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Temperature")
            top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k (0 disables)")
        run_btn = gr.Button("Generate", variant="primary")
        output = gr.Textbox(label="Output", lines=12)

        run_btn.click(
            fn=generate_until_eot,
            inputs=[prompt, max_new_tokens, temperature, top_k],
            outputs=[output],
        )
    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", share=True)
