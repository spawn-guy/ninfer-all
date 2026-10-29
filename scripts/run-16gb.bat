@echo off
setlocal DisableDelayedExpansion
rem Serve ONE model at a time on a 16 GB card. These are conservative, unmeasured defaults.
rem Usage: run-16gb.bat [gemma4|qwen38-27b] [--dry-run]
rem No argument opens a model chooser. --dry-run prints the command without loading anything.
rem
rem Qwen uses the smaller GSQ-RCO IQ3_S artifact, NOT qwen3_8_27b.ninfer.
rem Download it separately; see docs\maintainer\launcher-profiles.md.
rem Gemma keeps routed experts in pinned RAM; all model arithmetic still runs on the GPU.
rem Pinned RAM is not free under WDDM, even when the machine has abundant system RAM.
rem
rem Overrides: NINFER_MODEL_DIR, NINFER_SERVER, NINFER_HOST, NINFER_PORT,
rem NINFER_CONTEXT, NINFER_PREFILL_CHUNK, NINFER_HOST_CACHE_MIB,
rem NINFER_EXPERT_CACHE_MIB (Gemma only), NINFER_SPEC (none or mtp).
rem Vision, DFlash2 and the 24 GB launcher's large context/cache settings are not enabled.

set "MODEL_KEY=%~1"
if /i "%MODEL_KEY%"=="--help" goto :help
if /i "%MODEL_KEY%"=="-h" goto :help
if not "%~3"=="" goto :bad_arguments
if not "%~2"=="" if /i not "%~2"=="--dry-run" goto :bad_arguments
if not defined MODEL_KEY goto :choose_model

:resolve_model
if /i "%MODEL_KEY%"=="gemma4" goto :gemma4
if /i "%MODEL_KEY%"=="qwen38-27b" goto :qwen38
echo Unknown model: "%MODEL_KEY%" 1>&2
goto :bad_arguments

:choose_model
echo Serve one model on a 16 GB GPU:
echo   1  Gemma 4 QAT Q4_0, experts in RAM
echo   2  Qwen3.8-27B GSQ-RCO IQ3_S, separate artifact required
choice /c 12 /n /m "Choose 1 or 2: "
if errorlevel 255 exit /b 2
if errorlevel 2 goto :choose_qwen
if errorlevel 1 goto :choose_gemma
exit /b 2
:choose_qwen
set "MODEL_KEY=qwen38-27b"
goto :resolve_model
:choose_gemma
set "MODEL_KEY=gemma4"
goto :resolve_model

:gemma4
set "ARTIFACT=gemma-4-26B-A4B-it-QAT-Q4_0-MTP-ninfer-v3.ninfer"
set "MODEL_ID=gemma4"
set "FAMILY_ARGS=--expert-residency host"
set "EXPERT_CACHE=auto"
if defined NINFER_EXPERT_CACHE_MIB set "EXPERT_CACHE=%NINFER_EXPERT_CACHE_MIB%"
set "FAMILY_ARGS=%FAMILY_ARGS% --expert-cache-mib %EXPERT_CACHE%"
set "DRAFT_TOKENS=2"
goto :configure

:qwen38
set "ARTIFACT=Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer"
set "MODEL_ID=qwen3.8-27b"
set "FAMILY_ARGS=--gdn-state-fp16"
set "DRAFT_TOKENS=3"
goto :configure

:configure
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "MODEL_DIR=%ROOT%\models"
if defined NINFER_MODEL_DIR set "MODEL_DIR=%NINFER_MODEL_DIR%"
set "MODEL=%MODEL_DIR%\%ARTIFACT%"
set "SERVER=%ROOT%\build-windows\apps\Release\ninfer-serve.exe"
if defined NINFER_SERVER set "SERVER=%NINFER_SERVER%"
set "HOST=127.0.0.1"
set "PORT=8080"
set "CONTEXT=16384"
set "PREFILL_CHUNK=512"
set "HOST_CACHE=16384"
set "SPEC=none"
if defined NINFER_HOST set "HOST=%NINFER_HOST%"
if defined NINFER_PORT set "PORT=%NINFER_PORT%"
if defined NINFER_CONTEXT set "CONTEXT=%NINFER_CONTEXT%"
if defined NINFER_PREFILL_CHUNK set "PREFILL_CHUNK=%NINFER_PREFILL_CHUNK%"
if defined NINFER_HOST_CACHE_MIB set "HOST_CACHE=%NINFER_HOST_CACHE_MIB%"
if defined NINFER_SPEC set "SPEC=%NINFER_SPEC%"
set "SPEC_ARGS=--spec none"
if /i "%SPEC%"=="none" goto :launch
if /i not "%SPEC%"=="mtp" goto :bad_spec
set "SPEC_ARGS=--spec mtp --draft-tokens %DRAFT_TOKENS%"

:launch
if /i "%~2"=="--dry-run" goto :dry_run
if not exist "%SERVER%" goto :missing_server
if not exist "%MODEL%" goto :missing_model
echo Serving %MODEL_ID% at http://%HOST%:%PORT%/v1
echo One request, %CONTEXT% tokens, rk4v4 KV, %SPEC% speculation, text only.
echo Host context cache ceiling: %HOST_CACHE% MiB; Windows may clamp pinned allocations.
echo Stop this server before launching the other model.
"%SERVER%" "%MODEL%" --model-id %MODEL_ID% --host %HOST% --port %PORT% ^
  --max-concurrency 1 --max-context %CONTEXT% --kv-capacity %CONTEXT% ^
  --kv-dtype rk4v4 --prefill-chunk %PREFILL_CHUNK% --host-cache-mib %HOST_CACHE% ^
  %SPEC_ARGS% %FAMILY_ARGS%
exit /b %errorlevel%

:dry_run
echo "%SERVER%" "%MODEL%" --model-id %MODEL_ID% --host %HOST% --port %PORT% --max-concurrency 1 --max-context %CONTEXT% --kv-capacity %CONTEXT% --kv-dtype rk4v4 --prefill-chunk %PREFILL_CHUNK% --host-cache-mib %HOST_CACHE% %SPEC_ARGS% %FAMILY_ARGS%
exit /b 0

:missing_server
echo Missing server: "%SERVER%" 1>&2
echo Build ninfer-serve first or set NINFER_SERVER to its executable. 1>&2
exit /b 1

:missing_model
echo Missing model: "%MODEL%" 1>&2
if /i "%MODEL_KEY%"=="gemma4" goto :missing_gemma
if /i "%MODEL_KEY%"=="qwen38-27b" goto :missing_qwen
exit /b 1
:missing_gemma
echo Download the IQ3_S .ninfer artifact from: 1>&2
echo https://huggingface.co/WaveCut/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-NInfer-v3 1>&2
echo hf download WaveCut/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-NInfer-v3 gemma-4-26B-A4B-it-QAT-Q4_0-MTP-ninfer-v3.ninfer --local-dir models 1>&2
exit /b 1
:missing_qwen
echo Download the IQ3_S .ninfer artifact from: 1>&2
echo https://huggingface.co/WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3 1>&2
echo hf download WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3 Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer --local-dir models 1>&2
echo The original qwen3_8_27b.ninfer does not fit 16 GB; dense RAM offload is not supported. 1>&2
exit /b 1

:bad_spec
echo NINFER_SPEC must be none or mtp, got "%SPEC%". 1>&2
exit /b 2

:bad_arguments
call :help 1>&2
exit /b 2

:help
echo Usage: scripts\run-16gb.bat [gemma4^|qwen38-27b] [--dry-run]
echo Defaults: one request, 8192 tokens, rk4v4 KV, chunk 512, no speculation or vision.
echo Gemma: original QAT Q4_0 artifact, routed experts in pinned RAM with an automatic GPU cache.
echo Qwen: separate GSQ-RCO IQ3_S artifact; the original qwen3_8_27b.ninfer is too large.
echo Set NINFER_SPEC=mtp to try drafting; this needs additional VRAM.
exit /b 0
