import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bench import run_qwen_next_windows_5070ti as bench


class FlashNextWindowsBenchmarkTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows batch launcher")
    def test_launcher_exposes_measured_defaults_and_explicit_overrides(self):
        launcher = bench.ROOT / "scripts/run-qwen-next-16gb.bat"
        env = {key: value for key, value in os.environ.items() if key.upper() not in {
            "NINFER_CONTEXT", "NINFER_PREFILL_CHUNK", "NINFER_SPEC",
        }}

        def dry_run(overrides):
            return subprocess.run(
                f'"{os.environ["ComSpec"]}" /d /c call "{launcher}" iq3_s --dry-run',
                env={**env, **overrides}, capture_output=True, text=True, timeout=10,
            )

        result = dry_run({})
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--max-context 40960", "--kv-capacity 40960", "--prefill-chunk 2048",
                     "--kv-dtype bf16", "--spec none"):
            self.assertIn(flag, result.stdout)
        self.assertNotIn("--draft-tokens", result.stdout)
        result = dry_run({"NINFER_CONTEXT": "16384", "NINFER_PREFILL_CHUNK": "512",
                          "NINFER_SPEC": "mtp"})
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--max-context 16384", "--prefill-chunk 512", "--spec mtp",
                     "--draft-tokens 3", "--draft-min-p 0.5"):
            self.assertIn(flag, result.stdout)
        result = dry_run({"NINFER_SPEC": "invalid"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("NINFER_SPEC must be", result.stderr)

    def test_long_fixture_has_exact_budget_and_three_separated_keys(self):
        def encode(port, text):
            return list(text.encode("utf-8"))

        with patch.object(bench, "tokenize", side_effect=encode), patch.object(
            bench, "chat_prompt", side_effect=lambda port, text: "USER:" + text + "\nASSISTANT:"
        ):
            tokens = bench.long_prompt(8097, 40000)
        self.assertEqual(len(tokens), 40000)
        text = bytes(tokens).decode("utf-8", errors="replace")
        self.assertLess(text.index("LARCH-7291"), 512)
        self.assertGreater(text.index("COBALT-5836"), 19000)
        self.assertLess(text.index("COBALT-5836"), 21000)
        self.assertGreater(text.index("ORBIT-1947"), 39000)
        self.assertTrue(text.endswith("\nASSISTANT:"))

    def test_long_fixture_refuses_insufficient_budget(self):
        with patch.object(bench, "tokenize", side_effect=lambda port, text: list(text.encode())), \
                patch.object(bench, "chat_prompt", side_effect=lambda port, text: text):
            with self.assertRaises(ValueError):
                bench.long_prompt(8097, 10)

    def test_template_request_explicitly_disables_thinking(self):
        with patch.object(bench, "post", return_value={"prompt": "rendered"}) as post:
            self.assertEqual(bench.chat_prompt(8097, "hello"), "rendered")
        body = post.call_args.args[2]
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hello"}])

    def test_measure_checks_real_counts_and_reports_retrieval(self):
        from argparse import Namespace

        expected = {"alpha": "LARCH-7291", "beta": "COBALT-5836", "gamma": "ORBIT-1947"}
        record = {
            "event": "request_done",
            "result": {"completion_tokens": 64, "computed_prefill_tokens": 3, "prompt_tokens": 3},
            "timings_seconds": {"ttft": 2.5, "prefill": 2, "decode": 4},
        }
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "requests.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            with patch.object(bench, "post", return_value={"content": json.dumps(expected)}) as post:
                result = bench.measure(Namespace(port=8097, request_timeout=5), out,
                                       "long", [1, 2, 3], 64, True)
                self.assertEqual(result["decode_tps"], 16)
                self.assertEqual(result["prefill_tps"], 1.5)
                self.assertTrue(result["retrieval_exact"])
                self.assertEqual(result["retrieval_codes_present"], 3)
                with self.assertRaisesRegex(RuntimeError, "expected 65"):
                    bench.measure(Namespace(port=8097, request_timeout=5), out,
                                  "short", [1, 2, 3], 65, True)
                with self.assertRaisesRegex(RuntimeError, "exact token fixture"):
                    bench.measure(Namespace(port=8097, request_timeout=5), out,
                                  "long", [1, 2], 64, True)
                post.return_value = {"content": "```json\n" + json.dumps(expected) + "\n```"}
                fenced = bench.measure(Namespace(port=8097, request_timeout=5), out,
                                       "long", [1, 2, 3], 64, True)
                self.assertFalse(fenced["retrieval_exact"])
                self.assertEqual(fenced["retrieval_codes_present"], 3)

    def test_quality_fixture_includes_independent_domains_without_modifying_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            manifest = json.loads(bench.quality_fixture(out, 100).read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["streams"]), 3)
            self.assertEqual(manifest["modes"]["quick"], manifest["modes"]["full"])
            for stream in manifest["streams"]:
                self.assertEqual(len((out / stream["path"]).read_text(encoding="utf-8")), 100)
            self.assertEqual(len({stream["domain"] for stream in manifest["streams"]}), 3)

    def test_measure_refuses_response_without_completion_record(self):
        from argparse import Namespace

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "requests.jsonl").write_text('{"event":"server_start"}\n', encoding="utf-8")
            with patch.object(bench, "post", return_value={"content": "answer"}):
                with self.assertRaisesRegex(RuntimeError, "request_done"):
                    bench.measure(Namespace(port=8097, request_timeout=5), out,
                                  "short", "prompt", 64, True)


if __name__ == "__main__":
    unittest.main()
