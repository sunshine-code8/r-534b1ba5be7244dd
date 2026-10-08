"""Exercise live logging with real short-lived child processes, without point clouds."""

from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.wavedsp.evaluate import stream_command_to_report


class StreamReportTests(unittest.TestCase):
    def test_stdout_and_progress_are_saved_before_child_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, acknowledged = root / "logs/report.txt", root / "ack"
            case = self

            class LiveConsole(io.StringIO):
                def write(self, text: str) -> int:
                    count = super().write(text)
                    if "progress 1/2" in self.getvalue() and not acknowledged.exists():
                        # Child cannot finish until both terminal and file contain progress.
                        data = report.read_bytes()
                        case.assertIn(b"\rprogress 1/2", data)
                        case.assertNotIn(b"FINISHED", data)
                        acknowledged.touch()
                    return count

            child = '''
import sys, time
from pathlib import Path
print("START")
sys.stderr.write("\\rprogress 1/2")
sys.stderr.flush()
deadline = time.monotonic() + 5
while not Path(sys.argv[1]).exists():
    if time.monotonic() > deadline:
        raise RuntimeError("live output was not delivered")
    time.sleep(0.01)
print("\\nFINISHED")
'''
            terminal = LiveConsole()
            with patch("experiments.wavedsp.evaluate.sys.stdout", terminal):
                stream_command_to_report(
                    [sys.executable, "-u", "-c", child, str(acknowledged)], report)
            self.assertEqual(report.read_bytes().decode(), terminal.getvalue())
            self.assertIn("START", terminal.getvalue())
            self.assertIn("FINISHED", terminal.getvalue())

    def test_failure_preserves_stdout_and_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.txt"
            command = [sys.executable, "-u", "-c",
                       "import sys; print('partial'); print('failure', file=sys.stderr); sys.exit(7)"]
            terminal = io.StringIO()
            with patch("experiments.wavedsp.evaluate.sys.stdout", terminal):
                with self.assertRaises(subprocess.CalledProcessError) as raised:
                    stream_command_to_report(command, report)
            self.assertEqual(raised.exception.returncode, 7)
            self.assertEqual(report.read_text(), "partial\nfailure\n")
            self.assertEqual(report.read_text(), terminal.getvalue())

    def test_interruption_stops_child_and_keeps_partial_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.txt"
            children = []
            real_popen = subprocess.Popen

            def start_child(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                children.append(process)
                return process

            terminal = io.StringIO()
            with patch("experiments.wavedsp.evaluate.subprocess.Popen", side_effect=start_child), \
                 patch("experiments.wavedsp.evaluate.sys.stdout", terminal), \
                 patch.object(terminal, "write", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    stream_command_to_report(
                        [sys.executable, "-u", "-c",
                         "import time; print('started'); time.sleep(30)"], report)
            self.assertIsNotNone(children[0].poll())
            self.assertEqual(report.read_text(), "started\n")


if __name__ == "__main__":
    unittest.main()
