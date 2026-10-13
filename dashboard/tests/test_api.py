"""Dashboard API tests. The server is mock-backed and never shells out."""
import json
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from gpu_dashboard.actions import ActionError, Controller, validate_pdf
from gpu_dashboard.server import main, serve

ROOT = Path(__file__).resolve().parents[2]


class ControllerTests(unittest.TestCase):
    def test_evict_requires_confirm_and_clears_the_holder(self):
        ctl = Controller(mock=True, project_dir=str(ROOT))
        with self.assertRaises(ActionError):
            ctl.act("evict", {})
        job = ctl.act("evict", {"confirm": True})
        self.assertEqual(job["status"], "ok")
        self.assertIsNone(ctl.status()["holder"])
        self.assertEqual(ctl.status()["services"]["llama"]["state"], "stopped")

    def test_touch_clock_is_present_and_watchdog_reports_a_deadline(self):
        status = Controller(mock=True).status()
        self.assertIsNotNone(status["holder"]["last_active_at"])
        self.assertIsNotNone(status["watchdog"]["next_evict_in_s"])
        self.assertTrue(status["gpu"]["processes"])

    def test_llama_apply_rejects_shell_metacharacters(self):
        ctl = Controller(mock=True, project_dir=str(ROOT))
        with self.assertRaises(ActionError):
            ctl.act("llama-apply", {
                "model": "Qwen;rm -rf /",
                "lane": "upstream",
                "cache_k": "q4_0",
                "cache_v": "q4_0",
            })

    def test_llama_apply_switches_the_catalog_model(self):
        ctl = Controller(mock=True, project_dir=str(ROOT))
        job = ctl.act("llama-apply", {
            "model": "Qwen3.8-27B-UD-IQ2_M",
            "lane": "upstream",
            "cache_k": "q4_0",
            "cache_v": "q4_0",
        })
        self.assertEqual(job["status"], "ok")
        gateway = ctl.status()["gateway"]
        self.assertEqual(gateway["selected_model"], "Qwen3.8-27B-UD-IQ2_M")
        self.assertEqual(gateway["selected_runtime"], "upstream")
        self.assertEqual(gateway["context_length"], 262144)

    def test_pdf_path_rejects_traversal(self):
        with self.assertRaises(ActionError):
            validate_pdf("/tmp/../etc/passwd.pdf", must_exist=False)
        with self.assertRaises(ActionError):
            validate_pdf("scan.pdf", must_exist=False)
        self.assertEqual(validate_pdf("/tmp/scan.pdf", must_exist=False), "/tmp/scan.pdf")

    def test_real_actions_pass_an_argv_list(self):
        seen = []

        def runner(argv, timeout):
            seen.append((list(argv), timeout))
            return 0, "watchdog_stopped\n", ""

        ctl = Controller(mock=False, runner=runner, project_dir=str(ROOT))
        job = ctl.act("watchdog-stop", {})
        self.assertEqual(job["status"], "ok")
        self.assertEqual(seen[0][0], ["wingpu", "broker", "watchdog", "stop"])
        self.assertIsInstance(seen[0][0], list)

    def test_asr_start_puts_the_engine_in_argv(self):
        seen = []

        def runner(argv, timeout):
            seen.append(list(argv))
            return 0, "ok\n", ""

        ctl = Controller(mock=False, runner=runner)
        ctl.act("asr-start", {"engine": "qwen3"})
        self.assertEqual(seen[0], ["remote-asr", "start", "--engine", "qwen3"])

    def test_sources_do_not_invoke_a_shell(self):
        for relative in ("actions.py", "server.py"):
            text = (Path(__file__).resolve().parents[1] / "src" / "gpu_dashboard" / relative).read_text(encoding="utf-8")
            self.assertNotIn("shell=True", text)
            self.assertNotIn("os.system", text)
            self.assertNotIn("shell=True", text.replace("shell=False", ""))


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.httpd = serve("127.0.0.1", 0, Controller(mock=True, project_dir=str(ROOT)))
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def _req(self, path, *, method="GET", body=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                raw = response.read()
                return response.status, raw, response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), exc.headers.get("Content-Type", "")

    def test_health_and_page(self):
        status, raw, mime = self._req("/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(raw)["mock"])
        status, raw, mime = self._req("/")
        self.assertEqual(status, 200)
        self.assertIn(b"GPU Bridge", raw)
        self.assertNotIn(b"DRAFT", raw)
        self.assertIn("text/html", mime)

    def test_static_traversal_is_rejected(self):
        status, _raw, _mime = self._req("/static/../server.py")
        self.assertEqual(status, 404)

    def test_evict_and_logs_over_http(self):
        status, raw, _mime = self._req("/api/actions/evict", method="POST", body={})
        self.assertEqual(status, 400)
        status, raw, _mime = self._req("/api/actions/evict", method="POST", body={"confirm": True})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["job"]["status"], "ok")
        status, raw, _mime = self._req("/api/status")
        self.assertIsNone(json.loads(raw)["holder"])
        status, raw, _mime = self._req("/api/logs/llama?n=2")
        text = json.loads(raw)["text"]
        self.assertIn("tok/s", text)
        self.assertNotIn("load_model", text)

    def test_unknown_action(self):
        status, raw, _mime = self._req("/api/actions/sh", method="POST", body={"cmd": "id"})
        self.assertEqual(status, 404)
        self.assertIn(b"unknown", raw)

    def test_refuses_non_loopback_bind(self):
        self.assertEqual(main(["--host", "0.0.0.0", "--port", "9", "--mock"]), 2)
        with self.assertRaises(ActionError):
            serve("0.0.0.0", 9, Controller(mock=True))


if __name__ == "__main__":
    unittest.main()
