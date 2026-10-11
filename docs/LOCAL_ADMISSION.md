# Local request admission and drain

`serve/admission.py` counts resource requests before body parsing/tokenization. Local authenticated service clients (loopback, no Origin) can POST `/admin/drain` with `{"action":"drain"}`. New generation, load, VRAM adjustment and PDF requests then return HTTP 503 `server_draining`; existing requests continue. The response includes an owner token and active request count. POST `{"action":"resume","token":"..."}` releases only that owner's drain. GET `/admin/drain` reports state/counts without the token. Never stop the engine while active requests remain.

Optional `STRATA_ADMISSION_URL` points to a local memory admission service. Guarded requests require its JSON `ok:true`; timeout, failure or denial returns HTTP 503 `memory_admission`. Without this environment variable no external memory policy is applied. The local Spark Center deployment uses `http://127.0.0.1:11436/guard/admission` and a calibrated policy. This cannot guarantee CUDA allocations or stop an already running request from growing.

If the initial drain reply is lost, its owner may be unknown. Keep admission closed; inspect active_requests and /status before a deliberate service restart. Do not bypass the gate or start another model to recover it.

Validation: `python -m unittest serve.test_admission` uses a mock engine and exercises atomic admission races, owner checks, HTTP refusal/resume, cross-origin refusal and unavailable memory policy. The existing serve.test_server suite also ran (139 tests). On GB10, real drain returned 503 and resume allowed a real HTTP 200 completion; Spark Center switched to a local Ollama model and back. No full-memory stress test was performed.
