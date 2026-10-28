import os
import sys
import json
import time
import base64
import random
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from PIL import Image
from tqdm import tqdm
import openai
from openai import OpenAI
from google import genai
from google.genai import types, errors

sys.path.append("../Config")
from config import *


# Every pipeline script is called as: python3 <script>.py <DATASET> <MODE>
# MODE "test": runs on a small sample of train, everything goes to Outputs/<DATASET>/Sample/<pipeline>/ (for quality checks)
# MODE "full": runs on train and val, intermediate files go to Outputs/<DATASET>/Full/<pipeline>/, final data to Data/<DATASET>/ex/
# out_dir/cache/ keeps API responses (so interrupted runs resume), out_dir/logs/ keeps Gemini answers without an image
def get_args(pipeline):  # pipeline: "labels" or "inputs"
    dataset, mode = sys.argv[1], sys.argv[2]
    assert dataset in DATASETS, f"dataset must be one of {DATASETS}"
    assert mode in ["test", "full"], "mode must be 'test' or 'full'"
    out_dir = f"Outputs/{dataset}/{'Sample' if mode == 'test' else 'Full'}/{pipeline}/"
    os.makedirs(out_dir + "cache", exist_ok=True)
    print("=" * 30)
    print(f"script: {os.path.basename(sys.argv[0])}, dataset: {dataset}, mode: {mode}")
    return dataset, mode, out_dir


def splits_for(mode):
    return ["train"] if mode == "test" else AUG_SPLITS


def read_jsonl(path):
    if not os.path.exists(path):  # pandas would treat a missing path as a json string
        raise FileNotFoundError(path)
    return pd.read_json(path, lines=True, dtype={"id": str})  # keep ids as strings, e.g., "08291"


def save_jsonl(df, path):
    df.to_json(path, orient="records", lines=True, force_ascii=False)


# --- Parallel calls with a line-by-line cache, so long runs can be resumed ---
# items: list of (key, *args); fn(*args) -> str or None (None = failed, retried on the next run)
def parallel_cached(items, fn, cache_path, workers):
    done = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[r["key"]] = r["response"]
    todo = [it for it in items if it[0] not in done]
    print(f"    [cache] {len(items) - len(todo)} done, {len(todo)} to go ({cache_path})")

    with open(cache_path, "a", encoding="utf-8") as f:
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = {executor.submit(fn, *it[1:]): it[0] for it in todo}
        try:
            for fut in tqdm(as_completed(futures), total=len(futures)):
                key, response = futures[fut], fut.result()
                if response is not None:
                    done[key] = response
                    f.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")
                    f.flush()
        except Exception:
            executor.shutdown(cancel_futures=True)
            raise
        executor.shutdown()

    failed = sum(it[0] not in done for it in items)
    if failed:
        print(f"    [warn] {failed} calls failed; rerun the script to retry them.")
    return [done.get(it[0]) for it in items]


# --- OpenAI (gpt-5-nano) ---
openai_client = None  # created on first use, so scripts that don't call OpenAI run without a key


def encode_image(path):
    image = Image.open(path).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def OpenAI_model_response(prompt, image_path=None, model_name=GPT_MODEL):
    global openai_client
    if openai_client is None:
        openai_client = OpenAI(api_key=open(OPENAI_KEY_FILE, "r", encoding="utf-8").read().strip(), timeout=120, max_retries=3)

    content = []
    if image_path is not None:
        try:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}})
        except Exception as e:  # unreadable image: skip this meme instead of stopping the run
            print(f"    [image] {image_path}: {e}")
            return None
    content.append({"type": "text", "text": prompt})

    for attempt in range(5):
        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                reasoning_effort=GPT_REASONING_EFFORT,
            )
            return (response.choices[0].message.content or "").strip()
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError):
            raise  # wrong key or model name: stop instead of retrying every item
        except openai.BadRequestError as e:
            print(f"    [openai] bad request, skipping: {e}")
            return None
        except Exception as e:
            print(f"    [openai] {e}; sleep for 30 sec ({attempt + 1}/5).")
            time.sleep(30)
    return None


# --- Gemini (meme regeneration) ---
def Gemini_image_response(prompt, save_path, log_path):
    """Generates an image from a text prompt and saves it at save_path, in the format of its extension (.png or .jpg). Returns True if the image exists."""
    if os.path.exists(save_path):
        return True
    gemini_keys = [k.strip() for k in open(GEMINI_KEY_FILE, "r", encoding="utf-8") if k.strip()]
    if len(gemini_keys) == 0:
        raise ValueError(f"No Gemini keys found in {GEMINI_KEY_FILE}")

    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        safety_settings=[
            types.SafetySetting(category=c, threshold="BLOCK_NONE")
            for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ],
    )

    log_msg = ""
    for attempt in range(3):
        client = genai.Client(api_key=random.choice(gemini_keys))  # simple load balancer over keys
        try:
            response = client.models.generate_content(model=GEMINI_IMAGE_MODEL, contents=prompt, config=config)
            for candidate in response.candidates or []:
                for part in (candidate.content.parts if candidate.content and candidate.content.parts else []):
                    if part.inline_data and part.inline_data.data:
                        image = Image.open(BytesIO(part.inline_data.data))
                        if save_path.lower().endswith((".jpg", ".jpeg")):
                            image.convert("RGB").save(save_path, format="JPEG", quality=95)
                        else:
                            image.save(save_path, format="PNG")
                        return True
                    if part.text:
                        log_msg += part.text
            # answered without an image (possibly due to moderation), so don't retry
            finish = response.candidates[0].finish_reason if response.candidates else response.prompt_feedback
            log_msg += f"\n[no image] {finish}"
            break
        except errors.ClientError as e:
            if e.code in [401, 403, 404]:
                raise  # wrong key or model name
            log_msg += f"\n[client error] {e}"
            print(f"    [gemini] {e}; sleep for 60 sec ({attempt + 1}/3).")
            time.sleep(60)
        except Exception as e:
            log_msg += f"\n[error] {e}"
            print(f"    [gemini] {e}; sleep for 60 sec ({attempt + 1}/3).")
            time.sleep(60)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n{log_msg}\n")
    return False
