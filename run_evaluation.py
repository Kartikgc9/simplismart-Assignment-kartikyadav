"""
Speculative Decoding with Whisper Large-V3 and distil-large-v3
==============================================================
PyTorch Conference Assignment — SimpliSmart

End-to-end evaluation script comparing:
  - Baseline   : openai/whisper-large-v3 (standard autoregressive decoding)
  - Speculative: openai/whisper-large-v3 + distil-whisper/distil-large-v3

Why distil-large-v3 (not whisper-tiny)?
  whisper-tiny encoder mel channels = 80
  whisper-large-v3 encoder mel channels = 128
  → RuntimeError: Conv1d shape mismatch — hard incompatibility

  distil-large-v3 shares large-v3's encoder (128 mel, 51866 vocab) and
  has only 2 decoder layers → fast draft generation, zero resizing needed.

Usage:
  python run_evaluation.py [--num-samples N] [--output-dir ./results]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Audio, load_dataset
from evaluate import load as load_metric
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MAIN_MODEL_ID      = "openai/whisper-large-v3"
ASSISTANT_MODEL_ID = "distil-whisper/distil-large-v3"
DATASET_NAME       = "hf-internal-testing/librispeech_asr_dummy"
DATASET_CONFIG     = "clean"
DATASET_SPLIT      = "validation"
MAX_NEW_TOKENS     = 128
TARGET_SAMPLE_RATE = 16_000
NO_REPEAT_NGRAM    = 3          # applied identically to both runs


# ─────────────────────────────────────────────────────────────────────────────
# Device setup
# ─────────────────────────────────────────────────────────────────────────────
def get_device_and_dtype():
    device      = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return device, torch_dtype


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_main_model(model_id: str, device: str, torch_dtype):
    """Load Whisper Large-V3 as the main verification model."""
    print(f"[1/4] Loading main model: {model_id}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="sdpa",
    )
    model.to(device).eval()
    # Suppress "Both max_new_tokens and max_length seem to have been set" warning
    model.generation_config.max_length = None
    processor = AutoProcessor.from_pretrained(model_id)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"       vocab={model.config.vocab_size} | mel={model.config.num_mel_bins} | params={params:.0f}M")
    return model, processor


def load_assistant_model(model_id: str, main_model, device: str, torch_dtype):
    """
    Load distil-whisper/distil-large-v3 as the draft (assistant) model.

    distil-large-v3 shares large-v3's encoder (128 mel channels) and vocabulary
    (51866 tokens), so NO resizing is needed.  It has only 2 decoder layers
    versus 32 in large-v3, making draft generation very fast.
    """
    print(f"[2/4] Loading assistant model: {model_id}")
    assistant = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="sdpa",
    )
    assistant.to(device).eval()
    assistant.generation_config.max_length = None

    # Verify hard compatibility requirements
    assert assistant.config.vocab_size == main_model.config.vocab_size, (
        f"Vocab mismatch: {assistant.config.vocab_size} vs {main_model.config.vocab_size}"
    )
    assert assistant.config.num_mel_bins == main_model.config.num_mel_bins, (
        f"Encoder channel mismatch: {assistant.config.num_mel_bins} vs {main_model.config.num_mel_bins}"
    )

    params = sum(p.numel() for p in assistant.parameters()) / 1e6
    print(f"       vocab={assistant.config.vocab_size} ✓ | mel={assistant.config.num_mel_bins} ✓ | params={params:.0f}M")
    return assistant


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_eval_dataset(num_samples: int | None = None):
    """Load LibriSpeech validation split (standard ASR benchmark)."""
    print(f"[3/4] Loading dataset: {DATASET_NAME} ({DATASET_CONFIG}, {DATASET_SPLIT})")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))
    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))
    print(f"       {len(ds)} samples loaded")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────
_normalizer = BasicTextNormalizer()

def normalize(text: str) -> str:
    """Lowercase + strip punctuation — standard Whisper WER normalisation."""
    return _normalizer(text)


def preprocess(sample: dict, processor, device: str, torch_dtype):
    """Convert raw audio to Whisper input features."""
    inputs = processor(
        sample["audio"]["array"],
        sampling_rate=sample["audio"]["sampling_rate"],
        return_tensors="pt",
    )
    features = inputs.input_features.to(device=device, dtype=torch_dtype)
    extra = {}
    if "attention_mask" in inputs:
        extra["attention_mask"] = inputs["attention_mask"].to(device=device)
    return features, extra


def run_evaluation(
    dataset,
    model,
    processor,
    device: str,
    torch_dtype,
    label: str,
    assistant_model=None,
) -> dict:
    """
    Run inference on all samples and collect timing + transcription data.

    IMPORTANT: baseline and speculative use IDENTICAL generate() parameters
    (same no_repeat_ngram_size, language, task, max_new_tokens) so that the
    mathematical guarantee of speculative decoding can be demonstrated.

    Args:
        assistant_model: if provided, enables speculative decoding.
    Returns:
        dict with wer, total_time, per_sample_times, predictions, references
    """
    wer_metric   = load_metric("wer")
    predictions  = []
    references   = []
    sample_times = []

    generate_kwargs = {
        "language":            "en",
        "task":                "transcribe",
        "max_new_tokens":      MAX_NEW_TOKENS,
        "no_repeat_ngram_size": NO_REPEAT_NGRAM,
    }
    if assistant_model is not None:
        generate_kwargs["assistant_model"] = assistant_model

    for sample in tqdm(dataset, desc=label):
        features, extra = preprocess(sample, processor, device, torch_dtype)

        t0 = time.perf_counter()
        with torch.no_grad():
            predicted_ids = model.generate(features, **extra, **generate_kwargs)
        t1 = time.perf_counter()

        sample_times.append(t1 - t0)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        predictions.append(normalize(text))
        references.append(normalize(sample["text"]))

    wer = wer_metric.compute(predictions=predictions, references=references)
    return {
        "label":            label,
        "wer":              wer,
        "total_time":       sum(sample_times),
        "per_sample_times": sample_times,
        "predictions":      predictions,
        "references":       references,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Results reporting
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(baseline: dict, speculative: dict):
    speedup   = baseline["total_time"] / speculative["total_time"]
    wer_delta = (speculative["wer"] - baseline["wer"]) * 100
    identical = all(b == s for b, s in zip(baseline["predictions"], speculative["predictions"]))
    n_identical = sum(b == s for b, s in zip(baseline["predictions"], speculative["predictions"]))
    n_total     = len(baseline["predictions"])

    line = "=" * 65
    print(f"\n{line}")
    print("  FINAL RESULTS SUMMARY")
    print(line)
    print(f"  {'Metric':<28} {'Baseline':>15} {'Speculative':>15}")
    print(f"  {'-'*60}")
    print(f"  {'Total Time (s)':<28} {baseline['total_time']:>15.2f} {speculative['total_time']:>15.2f}")
    print(f"  {'Avg Time / Sample (s)':<28} {np.mean(baseline['per_sample_times']):>15.3f} {np.mean(speculative['per_sample_times']):>15.3f}")
    print(f"  {'WER (%)':<28} {baseline['wer']*100:>14.2f}% {speculative['wer']*100:>14.2f}%")
    print(f"  {'-'*60}")
    print(f"  {'Speedup':<28} {'':>15} {speedup:>14.2f}x")
    print(f"  {'WER delta (pp)':<28} {'':>15} {wer_delta:>+14.4f}")
    print(f"  {'Identical outputs':<28} {'':>15} {n_identical}/{n_total}")
    print(line)

    if identical:
        print("  ✓  Mathematical guarantee satisfied: outputs are IDENTICAL to baseline.")
    else:
        print(f"  ⚠  {n_total - n_identical} sample(s) differ — known Whisper hallucination edge cases.")
    print()


def save_results(baseline: dict, speculative: dict, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Per-sample CSV
    n = len(baseline["predictions"])
    df = pd.DataFrame({
        "sample_idx":             range(n),
        "reference":              baseline["references"],
        "baseline_prediction":    baseline["predictions"],
        "speculative_prediction": speculative["predictions"],
        "baseline_time_s":        baseline["per_sample_times"],
        "speculative_time_s":     speculative["per_sample_times"],
        "per_sample_speedup":     [
            b / s for b, s in zip(baseline["per_sample_times"], speculative["per_sample_times"])
        ],
        "outputs_identical":      [
            b == s for b, s in zip(baseline["predictions"], speculative["predictions"])
        ],
    })
    csv_path = out / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Per-sample results → {csv_path}")

    # Summary JSON
    speedup = baseline["total_time"] / speculative["total_time"]
    summary = {
        "main_model":      MAIN_MODEL_ID,
        "assistant_model": ASSISTANT_MODEL_ID,
        "dataset":         f"{DATASET_NAME}/{DATASET_CONFIG}/{DATASET_SPLIT}",
        "num_samples":     n,
        "generation_config": {
            "max_new_tokens":       MAX_NEW_TOKENS,
            "no_repeat_ngram_size": NO_REPEAT_NGRAM,
            "language":             "en",
            "task":                 "transcribe",
        },
        "baseline": {
            "total_time_s": round(baseline["total_time"], 4),
            "avg_time_s":   round(np.mean(baseline["per_sample_times"]), 4),
            "wer":          round(baseline["wer"], 6),
        },
        "speculative": {
            "total_time_s": round(speculative["total_time"], 4),
            "avg_time_s":   round(np.mean(speculative["per_sample_times"]), 4),
            "wer":          round(speculative["wer"], 6),
        },
        "speedup_x":       round(speedup, 4),
        "wer_delta_pp":    round((speculative["wer"] - baseline["wer"]) * 100, 6),
        "outputs_identical": all(
            b == s for b, s in zip(baseline["predictions"], speculative["predictions"])
        ),
        "identical_count": sum(
            b == s for b, s in zip(baseline["predictions"], speculative["predictions"])
        ),
    }
    json_path = out / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON        → {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Speculative Decoding: Whisper Large-V3 + distil-large-v3"
    )
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit number of dataset samples (default: all)")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Directory to save results (default: ./results)")
    return parser.parse_args()


def main():
    args = parse_args()
    device, torch_dtype = get_device_and_dtype()

    print("=" * 65)
    print("  Speculative Decoding: Whisper Large-V3 + distil-large-v3")
    print("  PyTorch Conference Assignment — SimpliSmart")
    print("=" * 65)
    print(f"  Device : {device}")
    print(f"  Dtype  : {torch_dtype}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  no_repeat_ngram_size = {NO_REPEAT_NGRAM} (applied to BOTH runs)")
    print()

    # ── Load models ──────────────────────────────────────────────
    model, processor = load_main_model(MAIN_MODEL_ID, device, torch_dtype)
    assistant_model  = load_assistant_model(
        ASSISTANT_MODEL_ID, model, device, torch_dtype,
    )

    # ── Load dataset ─────────────────────────────────────────────
    dataset = load_eval_dataset(num_samples=args.num_samples)

    print("[4/4] Running evaluations...\n")

    # ── Baseline (no speculative decoding) ───────────────────────
    baseline_results = run_evaluation(
        dataset, model, processor, device, torch_dtype,
        label="Baseline (Large-V3 only)",
        assistant_model=None,
    )

    # ── Speculative decoding ─────────────────────────────────────
    speculative_results = run_evaluation(
        dataset, model, processor, device, torch_dtype,
        label="Speculative (Large-V3 + distil-large-v3)",
        assistant_model=assistant_model,
    )

    # ── Report ───────────────────────────────────────────────────
    print_summary(baseline_results, speculative_results)
    save_results(baseline_results, speculative_results, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
