"""Quality-first Flash-Next comparisons through the public Windows products."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime


ROOT = Path(__file__).resolve().parents[2]
TABLE = "Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer"
MODELS = {
    key: f"Qwen3.8-Flash-Next-GSQ-RCO-{quant}-ninfer-v3.ninfer"
    for key, quant in (("iq3_s", "IQ3_S"), ("q2_0", "Q2_0"))
}
# One change from baseline, or from plain for the explicitly named plain arms.
PROFILES = {
    "baseline": {},
    "a16": {"extra": ["--no-prefill-a8"]},
    "plain": {"spec": "none"},
    "plain2048": {"spec": "none", "chunk": 2048},
    "mtp1": {"drafts": 1},
    "gpu-misses": {"dma": "1"},
    "chunk1024": {"chunk": 1024},
    "rk8v4": {"kv": "rk8v4"},
    "q2_0": {"model": "q2_0"},
    "q2-plain": {"model": "q2_0", "spec": "none"},
}
SHORT_PROMPTS = {
    "code": "Write a Python function that merges two sorted iterators lazily. "
            "Include type hints, handle duplicates and empty inputs, and explain its complexity.",
    "reasoning": "A tank has 120 litres. Each minute a pump removes 12% of the water "
                 "currently present and then adds 7 litres. Derive a closed form for the "
                 "amount after n minutes and calculate the limiting amount. Show your work.",
    "prose": "Explain how a database write-ahead log, checkpoints and backups work "
             "together after a power failure. Distinguish durability from replication.",
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def command_output(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def post(port: int, endpoint: str, body: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{endpoint}",
        json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode('utf-8')}") from error


def profile_options(name: str) -> dict:
    return {"model": "iq3_s", "kv": "bf16", "spec": "mtp", "drafts": 3,
            "dma": "auto", "chunk": 512, "extra": [], **PROFILES[name]}


def common_args(args: argparse.Namespace, options: dict) -> list[str]:
    return [
        "--ngram-table", str(args.bin_dir / "models" / TABLE),
        "--expert-residency", "disk", "--expert-cache-mib", "auto",
        "--expert-dma-share", options["dma"], "--ngram-residency", "disk",
        "--ngram-io", "buffered", "--ngram-ram-mib", "0",
        "--kv-dtype", options["kv"], "--kv-resident", "all", *options["extra"],
    ]


def wait_ready(process: subprocess.Popen, port: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with {process.returncode}; inspect stderr.log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    raise TimeoutError(f"Server not ready after {timeout}s")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)


def tokenize(port: int, text: str) -> list[int]:
    return post(port, "/tokenize", {"content": text})["tokens"]


def chat_prompt(port: int, content: str) -> str:
    return post(port, "/apply-template", {
        "model": "flash-next-bench", "messages": [{"role": "user", "content": content}],
        "reasoning_effort": "none",
    })["prompt"]


def long_prompt(port: int, count: int) -> list[int]:
    # Independent keys at the beginning, middle and end, with natural-text distractors.
    intro = ("Read the document and return only a JSON object with keys alpha, beta, gamma, "
             "containing the exact access codes assigned to those names. Ignore distractors.\n"
             "alpha access code: LARCH-7291.\n")
    middle = "\nbeta access code: COBALT-5836.\n"
    end = "\ngamma access code: ORBIT-1947.\nReturn the three access codes as JSON.\n"
    filler_path = ROOT / "eval/corpora/perplexity-heldout-2026-09/data/wikipedia_en/00.txt"
    filler = tokenize(port, filler_path.read_text(encoding="utf-8"))
    marker = "NINFER_DOCUMENT_5070TI"
    before, found, after = chat_prompt(port, marker).partition(marker)
    if not found:
        raise RuntimeError("Chat template omitted the document marker")
    head, mid, tail = (tokenize(port, text)
                       for text in (before + intro, middle, end + after))
    remaining = count - len(head) - len(mid) - len(tail)
    if remaining < 0 or len(filler) < remaining:
        raise ValueError("Long-context fixture cannot supply the requested token count")
    split = remaining // 2
    return head + filler[:split] + mid + filler[split:remaining] + tail


def measure(args: argparse.Namespace, out: Path, name: str, prompt: str | list[int],
            output_tokens: int, force_length: bool) -> dict:
    started = time.perf_counter()
    response = post(args.port, "/completion", {
        "prompt": prompt, "n_predict": output_tokens, "temperature": 0, "seed": 5070,
        "ignore_eos": force_length, "cache_prompt": False,
    }, args.request_timeout)
    elapsed = time.perf_counter() - started
    write_json(out / f"{name}.response.json", response)
    records = [json.loads(line) for line in (out / "requests.jsonl").read_text(
        encoding="utf-8").splitlines()]
    completed = [record for record in records if record.get("event") == "request_done"]
    if not completed:
        raise RuntimeError(f"{name}: server returned without a request_done record")
    done = completed[-1]
    write_json(out / f"{name}.record.json", done)
    result, timings = done["result"], done["timings_seconds"]
    generated, computed = result["completion_tokens"], result["computed_prefill_tokens"]
    if force_length and generated != output_tokens:
        raise RuntimeError(f"{name}: generated {generated}, expected {output_tokens}")
    if computed != result["prompt_tokens"]:
        raise RuntimeError(f"{name}: unexpected prefix reuse")
    if isinstance(prompt, list) and result["prompt_tokens"] != len(prompt):
        raise RuntimeError(f"{name}: prompt length does not match exact token fixture")
    row = {
        "name": name, "prompt_tokens": result["prompt_tokens"], "output_tokens": generated,
        "wall_seconds": elapsed, "ttft_seconds": timings["ttft"],
        "prefill_tps": computed / timings["prefill"] if timings["prefill"] else None,
        "decode_tps": generated / timings["decode"] if timings["decode"] else None,
        "speculative": done.get("speculative"), "timings": timings,
    }
    if name == "long":
        text = response["content"]
        try:
            parsed = json.loads(text.strip())
        except json.JSONDecodeError:
            parsed = None
        expected = {"alpha": "LARCH-7291", "beta": "COBALT-5836", "gamma": "ORBIT-1947"}
        row["retrieval_exact"] = parsed == expected
        row["retrieval_codes_present"] = sum(code in text for code in expected.values())
    return row


def run_serving(args: argparse.Namespace, name: str, out: Path) -> dict:
    options = profile_options(name)
    command = [
        str(args.bin_dir / "ninfer-serve.exe"),
        str(args.bin_dir / "models" / MODELS[options["model"]]),
        "--host", "127.0.0.1", "--port", str(args.port),
        "--model-id", "flash-next-bench", "--max-concurrency", "1",
        "--max-context", str(args.context), "--kv-capacity", str(args.context),
        "--prefill-chunk", str(options["chunk"]),
        "--no-prefix-reuse", "--spec", options["spec"], "--ngram-draft-tokens", "0",
        "--request-log-jsonl", str(out / "requests.jsonl"), *common_args(args, options),
    ]
    if options["spec"] == "mtp":
        command += ["--draft-tokens", str(options["drafts"]), "--draft-min-p", "0.5"]
    write_json(out / "command.json", command)
    rows = []
    with (out / "stdout.log").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
        started = time.monotonic()
        process = subprocess.Popen(command, cwd=args.bin_dir, stdout=stdout, stderr=stderr)
        try:
            wait_ready(process, args.port, args.startup_timeout)
            startup = time.monotonic() - started
            write_json(out / "gpu-ready.json", {"nvidia_smi": command_output([
                "nvidia-smi", "--query-gpu=name,memory.total,memory.used,pcie.link.gen.current,"
                "pcie.link.width.current", "--format=csv",
            ])})
            measure(args, out, "warmup", chat_prompt(args.port, "Explain what a CPU cache does."),
                    16, True)
            if args.phase == "long":
                prompt = long_prompt(args.port, args.prompt_tokens or (
                    args.context - args.output_tokens - 16))
                rows.append(measure(args, out, "long", prompt, args.output_tokens, False))
            else:
                for repeat in range(args.repetitions):
                    for kind, prompt in SHORT_PROMPTS.items():
                        rows.append(measure(args, out, f"{kind}-{repeat}",
                                            chat_prompt(args.port, prompt),
                                            args.output_tokens, True))
            summary = {"profile": name, "options": options, "phase": args.phase,
                       "startup_seconds": startup, "rows": rows}
            write_json(out / "summary.json", summary)
            return summary
        finally:
            stop_process(process)


def quality_fixture(out: Path, chars: int) -> Path:
    source = ROOT / "eval/corpora/perplexity-heldout-2026-09"
    streams = []
    for domain in ("wikipedia_en", "wikipedia_zh", "github_code"):
        filename = f"{domain}.txt"
        text = (source / "data" / domain / "00.txt").read_text(encoding="utf-8")[:chars]
        (out / filename).write_text(text, encoding="utf-8")
        streams.append({"id": domain, "domain": domain, "path": filename})
    manifest = out / "manifest.json"
    ids = [stream["id"] for stream in streams]
    write_json(manifest, {
        "corpus_id": f"heldout-prefix-{chars}-characters-per-domain",
        "streams": streams, "modes": {"quick": ids, "full": ids},
    })
    return manifest


def run_quality(args: argparse.Namespace, name: str, out: Path) -> dict:
    options = profile_options(name)
    manifest = quality_fixture(out, args.quality_chars)
    command = [
        str(args.bin_dir / "ninfer-perplexity.exe"),
        str(args.bin_dir / "models" / MODELS[options["model"]]),
        "--corpus", str(manifest), "--context", str(args.context),
        "--stride", str(args.context // 2), "--output", str(out / "ppl"),
        *common_args(args, options),
    ]
    write_json(out / "command.json", command)
    with (out / "stdout.log").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
        subprocess.run(command, cwd=args.bin_dir, stdout=stdout, stderr=stderr,
                       check=True, timeout=args.request_timeout)
    report = json.loads((out / "ppl" / "report.json").read_text(encoding="utf-8"))
    summary = {"profile": name, "phase": "quality", "report": report}
    write_json(out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=ROOT.parent / "ninfer-all_bin")
    parser.add_argument("--phase", choices=("short", "long", "quality"), default="short")
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES),
                        default=["baseline", "a16", "plain", "mtp1", "gpu-misses",
                                 "chunk1024", "rk8v4", "q2_0"])
    parser.add_argument("--context", type=int, default=40960)
    parser.add_argument("--output-tokens", type=int, default=96)
    parser.add_argument("--prompt-tokens", type=int,
                        help="exact long-phase prompt length; default fills the context budget")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--quality-chars", type=int, default=6000)
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.bin_dir = args.bin_dir.resolve()
    if args.context < 2048 or not 0 < args.output_tokens < args.context - 512:
        parser.error("context must be >=2048 with room for the output and a 512-token prompt")
    if args.repetitions < 1 or args.quality_chars < 1:
        parser.error("repetitions and quality-chars must be positive")
    if args.prompt_tokens is not None and (
        args.phase != "long" or not 512 <= args.prompt_tokens <= args.context - args.output_tokens - 16
    ):
        parser.error("prompt-tokens is for long only and must fit the context minus output and 16")
    if args.phase == "quality" and any(profile not in ("baseline", "a16", "rk8v4", "q2_0")
                                       for profile in args.profiles):
        parser.error("quality profiles: baseline, a16, rk8v4, q2_0 (no MTP in scoring)")
    required = [args.bin_dir / ("ninfer-perplexity.exe" if args.phase == "quality"
                                else "ninfer-serve.exe"), args.bin_dir / "models" / TABLE]
    required += [args.bin_dir / "models" / MODELS[profile_options(name)["model"]]
                 for name in args.profiles]
    for path in required:
        if not path.is_file():
            parser.error(f"missing prerequisite: {path}")
    if args.phase != "quality":
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
    output = (args.output or ROOT / "profiles" / "bench" / (
        "windows_5070ti_" + args.phase + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    hardware = {
        "gpu": command_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,"
                               "driver_version,pcie.link.gen.current,pcie.link.width.current",
                               "--format=csv"]),
        "system": command_output(["powershell.exe", "-NoProfile", "-Command",
                                  "Get-CimInstance Win32_Processor | Select-Object Name,"
                                  "NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json;"
                                  "Get-CimInstance Win32_OperatingSystem | Select-Object "
                                  "TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json"]),
        "binary": str(required[0]), "binary_size": required[0].stat().st_size,
        "binary_mtime": required[0].stat().st_mtime,
        "settings": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
        "cache_conditions": "OS file cache not evicted; sequential arms, warmup before timing. "
                            "GPU memory is system-wide, not process-exclusive. "
                            "Quality prefixes are a small screen, not full-corpus qualification.",
    }
    write_json(output / "environment.json", hardware)
    print(f"Results: {output}", flush=True)
    summaries, failures = [], []
    for name in args.profiles:
        print(f"Starting {args.phase}: {name}", flush=True)
        out = output / name
        out.mkdir()
        try:
            summary = (run_quality if args.phase == "quality" else run_serving)(args, name, out)
        except (RuntimeError, OSError, TimeoutError, subprocess.SubprocessError,
                KeyError, ValueError) as error:
            failure = {"profile": name, "error": f"{type(error).__name__}: {error}"}
            write_json(out / "failure.json", failure)
            failures.append(failure)
            print(f"FAILED {name}: {failure['error']}", flush=True)
        else:
            summaries.append(summary)
            if args.phase != "quality":
                rates = [row["decode_tps"] for row in summary["rows"]]
                print(f"{name}: median decode {statistics.median(rates):.2f} tok/s", flush=True)
            else:
                ppl = summary["report"]["overall"]["perplexity"]
                print(f"{name}: perplexity {ppl:.6f} (lower is better)", flush=True)
        write_json(output / "results.json", {"summaries": summaries, "failures": failures})
    if args.phase != "quality":
        with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["profile", "name", "prompt_tokens", "output_tokens", "wall_seconds",
                      "ttft_seconds", "prefill_tps", "decode_tps", "retrieval_exact",
                      "retrieval_codes_present"]
            writer = csv.DictWriter(handle, fields, extrasaction="ignore")
            writer.writeheader()
            for summary in summaries:
                for row in summary["rows"]:
                    writer.writerow({"profile": summary["profile"], **row})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
