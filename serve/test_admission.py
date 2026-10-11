"""Drain tests use the mock engine only; no model/GPU or production service."""
import json
import os
from pathlib import Path
import threading
import unittest
import urllib.request
import urllib.error
from unittest.mock import patch
from serve.admission import AdmissionGate
from serve.server import ByteTokenizer, MockEngine, Service, serve
from serve.frontend import ChatTemplate


class GateTests(unittest.TestCase):
    def test_race_has_no_unaccounted_request(self):
        for _ in range(100):
            gate = AdmissionGate()
            barrier = threading.Barrier(2)
            admitted = []
            def enter():
                barrier.wait()
                admitted.append(gate.enter())
            t = threading.Thread(target=enter)
            t.start(); barrier.wait()
            state = gate.drain()
            t.join()
            self.assertEqual(state['active_requests'], int(admitted[0]))
            self.assertFalse(gate.enter())

    def test_only_owner_can_resume(self):
        gate = AdmissionGate()
        token = gate.drain()['token']
        with self.assertRaises(ValueError): gate.resume('wrong')
        with self.assertRaises(ValueError): gate.drain()
        self.assertNotIn('token', gate.snapshot())
        gate.resume(token)
        self.assertTrue(gate.enter())
        gate.leave()
        self.assertEqual(gate.snapshot()['active_requests'], 0)


class HttpTests(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, '</think>\nOK', max_context=4096), tok,
                           ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = 'http://127.0.0.1:%s' % self.httpd.server_address[1]

    def post(self, path, body, headers=None):
        req = urllib.request.Request(self.base+path, data=json.dumps(body).encode(),
                                     headers={'Content-Type':'application/json', **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=3) as r:return r.status,json.load(r)
        except urllib.error.HTTPError as r:
            with r:return r.code,json.load(r)

    def test_drain_rejects_before_pdf_or_generation_and_resume_works(self):
        status, d = self.post('/admin/drain', {'action':'drain'})
        self.assertEqual(status, 200)
        for path in ['/documents/pdf','/v1/chat/completions','/load','/v1/load','/v1/vram']:
            self.assertEqual(self.post(path, {})[0], 503)
        self.assertEqual(self.post('/admin/drain', {'action':'resume','token':'bad'})[0],409)
        self.assertEqual(self.post('/admin/drain', {'action':'resume','token':d['token']})[0],200)
        self.assertEqual(self.post('/v1/chat/completions', {'messages':[{'role':'user','content':'hi'}], 'max_tokens':10})[0],200)

    def test_foreign_browser_cannot_drain(self):
        self.assertEqual(self.post('/admin/drain', {'action':'drain'}, {'Origin':'http://evil.test'})[0],403)
        self.assertFalse(self.svc.admission.snapshot()['draining'])

    def test_memory_probe_failure_closes_admission_without_leaking_active_count(self):
        with patch.dict(os.environ, {'STRATA_ADMISSION_URL':'http://127.0.0.1:1/unavailable'}):
            status, _ = self.post('/v1/chat/completions', {'messages':[]})
        self.assertEqual(status,503)
        self.assertEqual(self.svc.admission.snapshot()['active_requests'],0)


if __name__ == '__main__': unittest.main()
