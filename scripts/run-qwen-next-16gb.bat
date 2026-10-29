@echo off
setlocal DisableDelayedExpansion
rem Paths are relative to the working directory, not the location of this script.
rem Quality-first Flash-Next defaults; measured on a 5070 Ti with 96 GiB host RAM.
rem NINFER_CONTEXT, NINFER_PREFILL_CHUNK and NINFER_SPEC override the defaults.

if not "%~3"=="" goto :bad_arguments
if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h" goto :help
if not "%~2"=="" if /i not "%~2"=="--dry-run" goto :bad_arguments
set "MODEL_KEY=%~1"
if not defined MODEL_KEY goto :menu
goto :select_model

:menu
echo Qwen3.8-Flash-Next - Windows, 16 GiB VRAM / 64 GiB RAM
echo   1  Q2_0        model 38.49 GiB + shared table 26.92 GiB
echo   2  IQ3_S       model 54.50 GiB + shared table 26.92 GiB
echo   3  Coder IQ1_M model 31.02 GiB + shared table 26.92 GiB
choice /c 123 /n /m "Choose 1, 2 or 3: "
if errorlevel 255 exit /b 2
if errorlevel 3 goto :choose_coder
if errorlevel 2 goto :choose_iq3
if errorlevel 1 goto :choose_q2
exit /b 2

:choose_iq3
set "MODEL_KEY=iq3_s"
goto :select_model

:choose_coder
set "MODEL_KEY=coder"
goto :select_model

:choose_q2
set "MODEL_KEY=q2_0"

:select_model
if /i "%MODEL_KEY%"=="iq3_s" goto :iq3
if /i "%MODEL_KEY%"=="coder" goto :coder
if /i "%MODEL_KEY%"=="q2_0" goto :q2
goto :bad_arguments

:iq3
set "MODEL_REPO=WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3"
set "MODEL_FILE=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-ninfer-v3.ninfer"
set "MODEL_ID=qwen3.8-flash-next-iq3-s"
goto :configure

:coder
set "MODEL_REPO=WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-NInfer-v3"
set "MODEL_FILE=Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-ninfer-v3.ninfer"
set "MODEL_ID=qwen3.8-flash-next-coder"
goto :configure

:q2
set "MODEL_REPO=WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3"
set "MODEL_FILE=Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-ninfer-v3.ninfer"
set "MODEL_ID=qwen3.8-flash-next-q2-0"

:configure
set "TABLE_REPO=WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3"
set "TABLE_FILE=Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer"
set "MODEL_DIR=%CD%\models"
set "SERVER=%CD%\ninfer-serve.exe"
set "CONTEXT=40960"
set "PREFILL_CHUNK=2048"
set "SPEC=none"
if defined NINFER_CONTEXT set "CONTEXT=%NINFER_CONTEXT%"
if defined NINFER_PREFILL_CHUNK set "PREFILL_CHUNK=%NINFER_PREFILL_CHUNK%"
if defined NINFER_SPEC set "SPEC=%NINFER_SPEC%"
set "SPEC_ARGS=--spec none"
if /i "%SPEC%"=="none" goto :configured_spec
if /i not "%SPEC%"=="mtp" goto :bad_spec
set "SPEC_ARGS=--spec mtp --draft-tokens 3 --draft-min-p 0.5"

:configured_spec
set "RUN_ARGS=--host 127.0.0.1 --port 8080 --max-concurrency 1"
set "RUN_ARGS=%RUN_ARGS% --max-context %CONTEXT% --kv-capacity %CONTEXT% --kv-resident all --kv-dtype bf16"
set "RUN_ARGS=%RUN_ARGS% --prefill-chunk %PREFILL_CHUNK% --host-cache-mib 512"
set "RUN_ARGS=%RUN_ARGS% --expert-residency disk --expert-cache-mib auto --expert-dma-share auto"
set "RUN_ARGS=%RUN_ARGS% --ngram-residency disk --ngram-io buffered --ngram-ram-mib 0"
set "RUN_ARGS=%RUN_ARGS% %SPEC_ARGS%"
if /i "%~2"=="--dry-run" goto :dry_run
if not exist "%SERVER%" goto :missing_server

call :file_present "%MODEL_DIR%\%MODEL_FILE%"
set "NEED_MODEL=%errorlevel%"
call :file_present "%MODEL_DIR%\%TABLE_FILE%"
set "NEED_TABLE=%errorlevel%"
if "%NEED_MODEL%%NEED_TABLE%"=="00" goto :launch

echo.
echo Missing or empty artifacts:
if "%NEED_MODEL%"=="1" echo   %MODEL_FILE%
if "%NEED_TABLE%"=="1" echo   %TABLE_FILE%
echo Destination: "%MODEL_DIR%"
echo Use a local SSD, preferably NVMe. Both files are needed; the table is shared.
echo Total storage for model and table: IQ3_S 81.42 GiB, Coder 57.94 GiB, Q2_0 65.41 GiB.
echo Existing nonempty files are reused. Interrupted hf downloads can be resumed.
choice /c YN /n /m "Download the missing files with hf now? [Y/N]: "
if errorlevel 255 exit /b 2
if errorlevel 2 goto :cancelled
if not errorlevel 1 exit /b 2
where.exe hf >nul 2>&1
if errorlevel 1 goto :missing_hf
if not exist "%MODEL_DIR%\" mkdir "%MODEL_DIR%"
if not exist "%MODEL_DIR%\" goto :mkdir_failed
if "%NEED_MODEL%"=="1" call :download "%MODEL_REPO%" "%MODEL_FILE%"
if errorlevel 1 exit /b 1
if "%NEED_TABLE%"=="1" call :download "%TABLE_REPO%" "%TABLE_FILE%"
if errorlevel 1 exit /b 1

:launch
echo.
echo Serving %MODEL_ID% at http://127.0.0.1:8080/v1
echo One request, %CONTEXT%-token context, BF16 KV, %SPEC% speculation, text only.
echo Experts and n-gram rows stream from files through the Windows page cache.
echo The automatic GPU cache uses spare VRAM; CPU threads and work share are automatic.
echo IQ3_S or Q2_0 plus its table exceeds 64 GiB RAM: cold reads depend on SSD speed.
echo 5070 Ti measurements used 96 GiB host RAM; 64 GiB performance is not measured.
echo Coder is not measured. Close other GPU-heavy applications.
echo Press Ctrl+C to stop.
"%SERVER%" "%MODEL_DIR%\%MODEL_FILE%" --model-id %MODEL_ID% --ngram-table "%MODEL_DIR%\%TABLE_FILE%" %RUN_ARGS%
exit /b %errorlevel%

:file_present
if not exist "%~1" exit /b 1
if exist "%~1\" exit /b 1
for %%F in ("%~1") do if "%%~zF"=="0" exit /b 1
exit /b 0

:download
hf download "%~1" "%~2" --local-dir "%MODEL_DIR%"
if errorlevel 1 goto :download_failed
call :file_present "%MODEL_DIR%\%~2"
if errorlevel 1 goto :download_failed
exit /b 0

:download_failed
echo Download failed or did not produce a nonempty file: "%~2" 1>&2
echo Check free disk space, network access, and hf authentication, then rerun. 1>&2
exit /b 1

:dry_run
echo "%SERVER%" "%MODEL_DIR%\%MODEL_FILE%" --model-id %MODEL_ID% --ngram-table "%MODEL_DIR%\%TABLE_FILE%" %RUN_ARGS%
exit /b 0

:cancelled
echo Download cancelled. Server not started.
exit /b 1

:missing_server
echo Missing server: "%SERVER%" 1>&2
echo Run this launcher from the directory containing ninfer-serve.exe and its runtime DLLs. 1>&2
exit /b 1

:missing_hf
echo hf was not found on PATH. Install the Hugging Face CLI, then rerun. 1>&2
echo Windows installation: https://huggingface.co/docs/huggingface_hub/installation 1>&2
exit /b 1

:mkdir_failed
echo Could not create model directory: "%MODEL_DIR%" 1>&2
exit /b 1

:bad_spec
echo NINFER_SPEC must be none or mtp, got "%SPEC%". 1>&2
exit /b 2

:bad_arguments
call :help 1>&2
exit /b 2

:help
echo Usage: run-qwen-next-16gb.bat [iq3_s^|coder^|q2_0] [--dry-run]
echo No model argument opens a chooser. Coder uses the published IQ1_M artifact.
echo Run from the directory containing ninfer-serve.exe; files go in .\models.
echo --dry-run prints the launch command without checking files or downloading.
echo Missing or empty model/table files are downloaded with hf only after confirmation.
echo Existing nonempty files are not checksummed; the Engine validates artifact structure and binding.
echo Profile: disk experts/table, auto GPU cache and CPU share, 40K BF16 KV, no speculation or vision.
echo Overrides: NINFER_CONTEXT, NINFER_PREFILL_CHUNK, NINFER_SPEC=none or mtp.
echo Chunk 2048 favors long prompts; set NINFER_PREFILL_CHUNK=512 for short-chat comparisons.
exit /b 0