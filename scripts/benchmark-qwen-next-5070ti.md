I’ll inspect the existing benchmark and available hardware, then build a 5070 Ti version around a 40K context target. I’ll compare quality-preserving settings first and report speed separately from any quality tradeoffs.

The machine reports about **96 GiB RAM**, rather than the earlier 64 GiB target, and a PCIe link width of **x4**. That makes expert-transfer measurements especially important. I’ll test IQ3_S and Q2_0 at **40,960 tokens**, keep the n-gram table enabled, and use BF16 KV as the quality baseline.

The first screen shows that **MTP3 is substantially slower than plain decode** on this machine: 2.35–3.34 versus 4.72–7.22 tok/s across the three prompts. That’s a single pass, so I’ll recheck the finalists before choosing a setting. The 40K capacity fits; a full-length request and quality measurements are still pending.

The quality screen is complete. A16 prefill produced exactly the same perplexity as the default on these samples. RK8V4 raised aggregate perplexity by about **1.2%**: English improved, but Chinese and code worsened. I’ll keep BF16 for the quality-first recommendation and test larger prompt chunks before the full 40K request; short prompts cannot establish long-prompt prefill speed.

On the 8K-token test, 2,048-token chunks reduced time to first token from **203s to 139s**, but decode fell from **4.87 to 4.42 tok/s**. Both runs retrieved all three codes correctly. I’ll now test the full 40K window with 2,048-token chunks; the decode cost means this may suit long documents better than short chats.

The **40,816-token prompt completed** within the 40,960-token context: about **6 minutes to first token**, then **4.36 tok/s**. It returned all three correct code mappings, but wrapped the JSON in Markdown fences, so the strict-format check **failed**. I’m retaining that result and running a reversed-order short comparison before finalizing the defaults.

**Recommended: IQ3_S, BF16 KV, plain decoding, 40,960-token context, and 2,048-token prompt chunks.** I benchmarked the supplied Windows binary and updated `scripts\run-qwen-next-16gb.bat` with those defaults.

Added [`benchmark-qwen-next-5070ti.bat`](scripts/benchmark-qwen-next-5070ti.bat), backed by a Python 3.11 runner. It uses the provided server and perplexity executables—no compiled `bench` targets needed.

### Measured results

Your machine reports approximately **96 GiB RAM**, not 64 GiB, with an i7-9850H and **PCIe 3.0 x4 under load**. These results are specific to that setup.

| Setting | Finding |
|---|---|
| **Plain decode vs MTP3** | Plain was **1.85–2.16× faster** across six matched short-prompt observations, including a reversed-order recheck. |
| MTP1 | Better than MTP3, but still slower than plain decoding. |
| A16 prefill | Identical perplexity on the quality samples; short-request speed changes were mixed. |
| GPU-only misses / 1024-token chunks | Small, mixed single-pass differences; no established overall improvement. |
| **2048 vs 512-token chunks** | On an 8K prompt, TTFT fell **203 → 139 seconds**, but decode slowed **4.87 → 4.42 tok/s**. Larger chunks favor long documents, not necessarily short chats. |
| Q2_0 plain | Faster reasoning/prose decode, slower code decode; higher reasoning TTFT made that complete request slower than IQ3_S. |

The quality screen scored **7,833 held-out targets** across English, Chinese and code:

| Representation | Overall perplexity ↓ | Important exceptions |
|---|---:|---|
| **IQ3_S BF16** | **6.31292** | Recommended mixed/code baseline |
| IQ3_S RK8V4 | 6.38811 | +1.19% overall; English improved, Chinese/code worsened |
| Q2_0 BF16 | 7.20743 | +14.17% overall; code worsened **61.1%**, although English improved **11.9%** |

### Actual 40K execution

An exact **40,816-token prompt** completed within the 40,960-token allocation:

- **360 seconds to first token**, 113.4 prefill tok/s, **4.36 decode tok/s**.
- All three access-code mappings were correct.
- **Strict JSON formatting failed**: the model added Markdown fences. That failure remains recorded.
- Observed GPU use reached **15,516 MiB including the desktop**; this was not a process-only peak measurement.

To launch the updated quality-first profile:

```bat
cd /d D:\_open_source\ninfer-all_bin
D:\_open_source\ninfer-all\scripts\run-qwen-next-16gb.bat iq3_s
```

For mostly short chats, set `NINFER_PREFILL_CHUNK=512` first. Context and speculation are also overridable through `NINFER_CONTEXT` and `NINFER_SPEC`.

Full tables, methods and limitations are in [`bench\README.md`](bench/README.md); raw evidence is under `profiles\bench\5070ti_*`. Initial runner setup errors were corrected before successful measurements. The actual launcher also served successfully with its prefix cache enabled, though the throughput benchmarks disabled prefix reuse. **Coder, 64 GiB host performance, full-window Q2_0, broad task accuracy and long-context perplexity remain unverified.**