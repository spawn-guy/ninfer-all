@echo off
setlocal DisableDelayedExpansion
rem Uses the public HTTP server and offline perplexity evaluator; no bench build needed.
rem Override NINFER_BENCH_BIN_DIR for another native release directory.
if not defined NINFER_BENCH_BIN_DIR set "NINFER_BENCH_BIN_DIR=D:\_open_source\ninfer-all_bin"
where.exe py >nul 2>&1
if errorlevel 1 (
  echo Python 3.11 and the py launcher are required. 1>&2
  exit /b 1
)
py -3.11 "%~dp0..\tools\bench\run_qwen_next_windows_5070ti.py" --bin-dir "%NINFER_BENCH_BIN_DIR%" %*
exit /b %errorlevel%
