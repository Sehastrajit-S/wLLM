"""Wraps run_server.py as an installable Windows Service, so wLLM starts
on boot and survives logoff -- pywin32's standard ServiceFramework pattern.
uvicorn.run() blocks its calling thread and offers no external "stop"
signal, so the actual server runs on a background thread while the service's
main thread just waits on SvcStop's event; stopping asks uvicorn's own
Server instance to exit (its documented shutdown mechanism) rather than
killing the thread outright.

Install (as Administrator, from an activated venv so pywin32 is on the path):
    python scripts/wllm_service.py --startup auto install
    python scripts/wllm_service.py start
Configure the model/flags by editing WLLM_SERVICE_ARGS below, then
reinstalling (or via `python scripts/wllm_service.py update` after editing
and re-running install). Uninstall with:
    python scripts/wllm_service.py stop
    python scripts/wllm_service.py remove

Requires pywin32 (`pip install wllm[service]` or `pip install pywin32`),
which is Windows-only and therefore not a default dependency of this
project -- see pyproject.toml's `service` extra.
"""
import sys
import threading

sys.path.insert(0, "src")

# Args passed to run_server.py's argument parser, as if typed on the command
# line -- edit this list for your deployment (model, port, flags), then
# reinstall the service for changes to take effect.
WLLM_SERVICE_ARGS = [
    "--model", "Qwen/Qwen2.5-0.5B-Instruct",
    "--host", "0.0.0.0",
    "--port", "8000",
]

try:
    import servicemanager
    import win32event
    import win32service
    import win32serviceutil
except ImportError:
    print("pywin32 is required to run this as a Windows Service: pip install pywin32", file=sys.stderr)
    raise


class wLLMService(win32serviceutil.ServiceFramework):
    _svc_name_ = "wLLM"
    _svc_display_name_ = "wLLM Inference Server"
    _svc_description_ = "OpenAI-compatible LLM inference server (wLLM)."

    def __init__(self, args):
        super().__init__(args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self._uvicorn_server = None

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        win32event.SetEvent(self.stop_event)

    def SvcDoRun(self):
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE, servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, ""),
        )
        server_thread = threading.Thread(target=self._run_server, daemon=True)
        server_thread.start()
        win32event.WaitForSingleObject(self.stop_event, win32event.INFINITE)
        server_thread.join(timeout=30)

    def _run_server(self):
        import argparse

        import torch
        import uvicorn

        from wllm.api.server import create_app
        from wllm.baseline.model import DEFAULT_MODEL_ID

        # Mirrors run_server.py's parser -- kept in sync manually since a
        # Windows Service has no terminal to pass real argv through, only
        # this file's WLLM_SERVICE_ARGS list.
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default=DEFAULT_MODEL_ID)
        parser.add_argument("--gguf", default=None)
        parser.add_argument("--tokenizer", default=None)
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--num-blocks", type=int, default=256)
        parser.add_argument("--block-size", type=int, default=16)
        parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
        parser.add_argument("--cuda-graphs", action="store_true")
        parser.add_argument("--prefix-caching", action="store_true")
        parser.add_argument("--chunked-prefill-tokens", type=int, default=None)
        parser.add_argument("--cpu-swap", action="store_true")
        parser.add_argument("--speculative-decoding", action="store_true")
        parser.add_argument("--speculative-ngram-size", type=int, default=3)
        parser.add_argument("--speculative-max-draft-len", type=int, default=4)
        parser.add_argument("--lora-modules", nargs="*", default=None)
        parser.add_argument("--api-key", dest="api_keys", nargs="*", default=None)
        parser.add_argument("--rate-limit-rpm", type=int, default=None)
        parser.add_argument("--log-level", default="INFO")
        args = parser.parse_args(WLLM_SERVICE_ARGS)

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
        lora_modules = dict(entry.split("=", 1) for entry in args.lora_modules) if args.lora_modules else None
        app = create_app(
            args.model,
            num_blocks=args.num_blocks,
            block_size=args.block_size,
            dtype=dtype,
            gguf_path=args.gguf,
            tokenizer_id=args.tokenizer,
            use_cuda_graphs=args.cuda_graphs,
            enable_prefix_caching=args.prefix_caching,
            max_prefill_tokens_per_step=args.chunked_prefill_tokens,
            enable_cpu_swap=args.cpu_swap,
            enable_speculative_decoding=args.speculative_decoding,
            speculative_ngram_size=args.speculative_ngram_size,
            speculative_max_draft_len=args.speculative_max_draft_len,
            lora_modules=lora_modules,
            api_keys=args.api_keys,
            rate_limit_per_minute=args.rate_limit_rpm,
            log_level=args.log_level,
        )

        config = uvicorn.Config(app, host=args.host, port=args.port, log_config=None)
        self._uvicorn_server = uvicorn.Server(config)
        self._uvicorn_server.run()


if __name__ == "__main__":
    win32serviceutil.HandleCommandLine(wLLMService)
