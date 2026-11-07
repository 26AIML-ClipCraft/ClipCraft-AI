"""Runs a local Qwen VL (Qwen2.5-VL or Qwen3-VL) on a few frames and prints {"query": "<one sentence>"}. Standalone on purpose: it needs a newer transformers
(>= 4.57 for Qwen3-VL) than this project pins (4.41.2), so query_gen.py starts it with the python of another environment.

  python qwen_describe.py --model Qwen/Qwen3-VL-8B-Instruct --frames a.jpg b.jpg ...
"""
import argparse
import json

PROMPT = ("These are frames sampled in order from one video. Write ONE short English sentence (at most 20 words) saying what "
          "the video is about, like a video title. Reply with the sentence only.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    a = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    proc = AutoProcessor.from_pretrained(a.model)           # frames arrive already scaled to <= 640 px
    images = [Image.open(f).convert("RGB") for f in a.frames]
    messages = [{"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=images, return_tensors="pt").to(dev)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
    sentence = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    print(json.dumps({"query": " ".join(sentence.split())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
