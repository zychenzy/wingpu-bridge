"""Tests for two fixes:

1. Qwen3.8 reasoning defaults are config-driven and reach the llama-server launch args.
2. A failing remote/build command raises an error that actually shows what went wrong
   (exit code plus the tail of both output streams) instead of swallowing it.
"""
import subprocess
import unittest
from unittest import mock

from wingpu_cli.main import (
    OUTPUT_TAIL_LINES,
    RuntimeDefaults,
    WingpuError,
    build_runtime,
    load_settings,
    reasoning_server_args,
    run,
    start_remote_runtime,
    tail_text,
    validate_extra_server_args,
    validate_reasoning_effort,
)
import wingpu_cli.main as main

from test_gateway_recovery import make_settings


def completed(returncode=1, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=["ssh", "host"], returncode=returncode, stdout=stdout, stderr=stderr)


class CommandFailureOutputTests(unittest.TestCase):
    """Fix 2: build failures must be loud."""

    def test_failure_message_carries_exit_code_and_both_streams(self):
        with mock.patch.object(main.subprocess, "run", return_value=completed(2, "on stdout", "on stderr")):
            with self.assertRaises(WingpuError) as ctx:
                run(["ssh", "host", "false"])
        message = str(ctx.exception)
        self.assertIn("exit code 2", message)
        self.assertIn("on stdout", message)
        self.assertIn("on stderr", message)

    def test_real_error_on_stdout_is_not_hidden_by_warnings_on_stderr(self):
        """The regression: a ninja/npm error goes to stdout while cmake writes warnings to
        stderr, and only stderr used to be reported, so the actual cause was invisible."""
        stdout = "\n".join([
            "[1/2] Building CXX object",
            "FAILED: vendor/webui",
            "npm: not found (ENOENT) via WSL interop",
            "ninja: build stopped: subcommand failed.",
        ])
        stderr = "CMake Warning: Manually-specified variables were not used: LLAMA_WEBUI_HF_BUCKET"
        with mock.patch.object(main.subprocess, "run", return_value=completed(1, stdout, stderr)):
            with self.assertRaises(WingpuError) as ctx:
                run(["ssh", "host", "build"])
        message = str(ctx.exception)
        self.assertIn("npm: not found (ENOENT) via WSL interop", message)
        self.assertIn("ninja: build stopped", message)
        self.assertIn("LLAMA_WEBUI_HF_BUCKET", message)

    def test_output_tail_is_bounded_and_marked_as_truncated(self):
        noisy = "\n".join(f"line {i}" for i in range(500))
        with mock.patch.object(main.subprocess, "run", return_value=completed(1, noisy, "")):
            with self.assertRaises(WingpuError) as ctx:
                run(["ssh", "host", "noisy"])
        message = str(ctx.exception)
        self.assertIn("line 499", message)          # the tail, where the real error lives
        self.assertNotIn("line 100", message)       # the head is dropped
        self.assertIn("truncated", message)
        self.assertLessEqual(len(message.splitlines()), OUTPUT_TAIL_LINES + 6)

    def test_failure_with_no_output_still_reports_the_exit_code(self):
        with mock.patch.object(main.subprocess, "run", return_value=completed(3, "", "")):
            with self.assertRaises(WingpuError) as ctx:
                run(["ssh", "host", "silent"])
        self.assertIn("exit code 3", str(ctx.exception))

    def test_timeout_reports_partial_output(self):
        timeout_exc = subprocess.TimeoutExpired(cmd=["ssh"], timeout=5, output="partial stdout", stderr="partial stderr")
        with mock.patch.object(main.subprocess, "run", side_effect=timeout_exc):
            with self.assertRaises(WingpuError) as ctx:
                run(["ssh", "host", "slow"], timeout=5)
        message = str(ctx.exception)
        self.assertIn("timed out", message)
        self.assertIn("partial stdout", message)
        self.assertIn("partial stderr", message)

    def test_check_false_does_not_raise(self):
        with mock.patch.object(main.subprocess, "run", return_value=completed(1, "out", "err")):
            self.assertEqual(run(["ssh", "host", "false"], check=False).returncode, 1)

    def test_build_runtime_surfaces_the_remote_failure_tail(self):
        """wingpu build goes through run_wsl_script -> run, so the tail must reach the user."""
        stdout = "\n".join(["configuring...", "FAILED: bin/llama-server", "ninja: build stopped: subcommand failed."])
        settings = make_settings()
        with mock.patch.object(main.subprocess, "run", return_value=completed(1, stdout, "CMake Warning: unused variable")):
            with self.assertRaises(WingpuError) as ctx:
                build_runtime(settings, "upstream")
        message = str(ctx.exception)
        self.assertIn("exit code 1", message)
        self.assertIn("ninja: build stopped", message)
        self.assertIn("CMake Warning", message)

    def test_tail_text_handles_empty_and_bytes(self):
        self.assertEqual(tail_text(None), "")
        self.assertEqual(tail_text(""), "")
        self.assertEqual(tail_text(b"a\nb"), "a\nb")


class ReasoningConfigTests(unittest.TestCase):
    """Fix 1: reasoning_effort / reasoning_budget are config-driven."""

    def test_no_flags_when_unset(self):
        self.assertEqual(reasoning_server_args(RuntimeDefaults()), [])

    def test_effort_flag_is_emitted(self):
        defaults = RuntimeDefaults(reasoning_effort="medium")
        self.assertEqual(reasoning_server_args(defaults), ["--reasoning-effort", "medium"])

    def test_budget_flag_is_emitted_only_when_non_negative(self):
        self.assertEqual(reasoning_server_args(RuntimeDefaults(reasoning_budget=-1)), [])
        self.assertEqual(reasoning_server_args(RuntimeDefaults(reasoning_budget=64)), ["--reasoning-budget", "64"])
        # 0 is a no-op on llama.cpp build 10454, but it is the user's choice to pass it.
        self.assertEqual(reasoning_server_args(RuntimeDefaults(reasoning_budget=0)), ["--reasoning-budget", "0"])

    def test_both_flags_together(self):
        defaults = RuntimeDefaults(reasoning_effort="low", reasoning_budget=128)
        self.assertEqual(
            reasoning_server_args(defaults),
            ["--reasoning-effort", "low", "--reasoning-budget", "128"],
        )

    def test_launch_script_passes_the_configured_effort(self):
        settings = make_settings()
        settings.runtime_defaults.reasoning_effort = "medium"
        captured = {}

        def fake_run_wsl_script(_settings, script, **_kwargs):
            captured["script"] = script
            return completed(0)

        with mock.patch.object(main, "run_wsl_script", side_effect=fake_run_wsl_script), \
             mock.patch.object(main, "catalog_entry", return_value={"context_length": 65536}), \
             mock.patch.object(main, "remote_model_path", return_value="/models/qwen.gguf"):
            start_remote_runtime(settings, "upstream-mtp", "Qwen3.8-27B-UD-Q3_K_XL", "q4_0", "q4_0", True)

        script = captured["script"]
        self.assertIn("--reasoning-effort medium", script)
        self.assertNotIn("--reasoning-budget", script)
        # the lane's own args must survive alongside the reasoning flags
        self.assertIn("--spec-type draft-mtp", script)

    def test_launch_script_omits_reasoning_flags_by_default(self):
        settings = make_settings()
        captured = {}

        with mock.patch.object(main, "run_wsl_script", side_effect=lambda _s, script, **_k: captured.setdefault("script", script)), \
             mock.patch.object(main, "catalog_entry", return_value={"context_length": 65536}), \
             mock.patch.object(main, "remote_model_path", return_value="/models/qwen.gguf"):
            start_remote_runtime(settings, "upstream", "Qwen3.8-27B-UD-Q3_K_XL", "q4_0", "q4_0", True)

        self.assertNotIn("--reasoning", captured["script"])

    def test_unsupported_effort_is_rejected_up_front(self):
        # 'meduim' is a typo and 'none' is not a template level (it is a per-request field),
        # both of which would otherwise turn every chat request into an HTTP 500.
        for bad in ["meduim", "none", "off"]:
            with self.subTest(value=bad):
                with self.assertRaisesRegex(WingpuError, "unsupported value"):
                    validate_reasoning_effort(bad)

    def test_supported_efforts_and_empty_are_accepted(self):
        for good in ["", "default", "minimal", "low", "medium", "high", "xhigh", "max"]:
            with self.subTest(value=good):
                validate_reasoning_effort(good)

    def test_reasoning_flags_are_rejected_in_extra_server_args(self):
        for flag in ["--reasoning-effort", "--reasoning-budget"]:
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(WingpuError, "managed by wingpu"):
                    validate_extra_server_args([flag, "medium"], source_label="runtime_defaults.extra_server_args")

    def test_shipped_defaults_enable_medium_effort(self):
        """The committed default is the mitigation for Qwen3.8 spending a small max_tokens
        budget entirely on thinking; a regression here silently restores the old behavior."""
        defaults_toml = main.read_config_bytes("wingpu.defaults.toml").decode("utf-8")
        self.assertIn('reasoning_effort = "medium"', defaults_toml)
        self.assertIn("reasoning_budget = -1", defaults_toml)

    def _load_with_override(self, local_override: bytes):
        # Read the shipped defaults before Path.open is mocked out from under read_config_bytes.
        defaults = main.read_config_bytes("wingpu.defaults.toml")
        with mock.patch.object(main, "read_config_bytes", return_value=defaults), \
             mock.patch.object(main, "project_config_path", return_value=main.Path("/tmp/wingpu.local.toml")), \
             mock.patch("pathlib.Path.exists", return_value=True), \
             mock.patch("pathlib.Path.open", mock.mock_open(read_data=local_override)):
            return load_settings()

    def test_load_settings_rejects_a_bad_reasoning_effort(self):
        override = b'[connection]\nhost = "win-gpu"\n[runtime_defaults]\nreasoning_effort = "meduim"\n'
        with self.assertRaisesRegex(WingpuError, "unsupported value"):
            self._load_with_override(override)

    def test_load_settings_reads_reasoning_keys_from_the_shipped_defaults(self):
        settings = self._load_with_override(b'[connection]\nhost = "win-gpu"\n')
        self.assertEqual(settings.runtime_defaults.reasoning_effort, "medium")
        self.assertEqual(settings.runtime_defaults.reasoning_budget, -1)


if __name__ == "__main__":
    unittest.main()
