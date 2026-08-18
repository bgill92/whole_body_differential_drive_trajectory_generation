#!/usr/bin/env python3
"""Regenerate assets/demo.gif from the project's own trajectory output.

Runs the pipeline with WBDD_RRD_PATH set so the Rerun recording is written to
a file instead of spawning a viewer, plays that recording back in a headless
Rerun viewer (`rerun --headless`), steps the `step` timeline while
screenshotting each knot via the viewer's MCP server (`rerun viewer-mcp`), and
assembles a looping GIF with ffmpeg. Requires `cargo`, `rerun` (rerun-cli /
rerun-sdk, same version as the crate's rerun dependency) and `ffmpeg` on PATH.
"""

import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 9877
STEP_FIRST, STEP_LAST, STEP_STRIDE = 0, 100, 2
# Headless viewer screen is 1920x1080; crop away the top title/view bars and
# the bottom time panel, keeping the 3D viewport.
CROP = "crop=1920:988:0:58"


def require(name):
    path = shutil.which(name)
    if not path:
        sys.exit(f"render_demo: {name} not found on PATH")
    return path


class Mcp:
    """Minimal JSON-RPC client for `rerun viewer-mcp` over stdio."""

    def __init__(self, rerun):
        self.proc = subprocess.Popen(
            [rerun, "viewer-mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._id = 0
        self.call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "render_demo", "version": "0.1"},
            },
        )
        self.call("notifications/initialized", notify=True)

    def call(self, method, params=None, notify=False):
        self._id += 1
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            msg["id"] = self._id
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        if notify:
            return None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("viewer-mcp exited")
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                continue
            if resp.get("id") == self._id:
                if "error" in resp:
                    raise RuntimeError(f"{method}: {resp['error']}")
                return resp.get("result")

    def tool(self, name, args=None):
        res = self.call("tools/call", {"name": name, "arguments": args or {}})
        return "\n".join(
            p.get("text", "") for p in res.get("content", []) if p.get("type") == "text"
        )

    def button(self, label):
        """First button whose accessible label contains `label`."""
        data = json.loads(
            self.tool("query_tree", {"role": "button", "label_contains": label})
        )
        nodes = data["nodes"] if isinstance(data, dict) else data
        if not nodes:
            raise RuntimeError(f"no button matching {label!r}")
        return nodes[0]["id"]

    def close(self):
        self.proc.terminate()


def main():
    cargo, rerun, ffmpeg = (require(n) for n in ("cargo", "rerun", "ffmpeg"))
    work = tempfile.mkdtemp(prefix="wbdd-demo-")
    rrd = os.path.join(work, "demo.rrd")
    frames = os.path.join(work, "frames")
    os.makedirs(frames)

    print("solving trajectory and recording to", rrd)
    subprocess.run(
        [cargo, "run", "--release"],
        cwd=ROOT,
        env={**os.environ, "WBDD_RRD_PATH": rrd},
        check=True,
    )

    # Own session so the cleanup below also reaches the real viewer binary,
    # which `rerun` shims (e.g. the rerun-sdk pip entry point) spawn as a child.
    viewer = subprocess.Popen(
        [
            rerun,
            "--headless",
            "--hide-welcome-screen",
            "--port",
            str(PORT),
            rrd,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        mcp = Mcp(rerun)
        try:
            def ok(text):
                try:
                    return json.loads(text).get("ok") is True
                except json.JSONDecodeError:
                    return False

            endpoint = f"rerun+http://127.0.0.1:{PORT}/proxy"
            deadline = time.time() + 60
            while time.time() < deadline:
                if ok(mcp.tool("connect", {"endpoint": endpoint})) and (
                    '"application_id":"urdf_view"' in mcp.tool("viewer_state")
                ):
                    break
                time.sleep(1)
            else:
                sys.exit("render_demo: viewer never loaded the recording")

            # Full-screen 3D view: maximize it, hide the side/time panels, and
            # park the pointer on empty sky so no hover tooltip is captured.
            mcp.tool("click", {"id": mcp.button("Maximize view")})
            for label in ("Blueprint panel toggle", "Selection panel toggle", "Time panel toggle"):
                mcp.tool("click", {"id": mcp.button(label)})
            mcp.tool("hover", {"pos": [60, 60]})
            # Zoom in a touch from the default scene fit.
            mcp.tool("scroll", {"pos": [960, 540], "delta": [0, -2]})
            mcp.tool("hover", {"pos": [60, 60]})
            time.sleep(1)

            for step in range(STEP_FIRST, STEP_LAST + 1, STEP_STRIDE):
                mcp.tool("set_time", {"timeline": "step", "time": step, "play": False})
                time.sleep(0.25)
                mcp.tool(
                    "screenshot",
                    {
                        "save_path": f"{frames}/frame_{step:04d}.png",
                        "pixels_per_point": 1.0,
                    },
                )
                print(f"captured step {step}")
        finally:
            mcp.close()
    finally:
        # SIGTERM the whole process group; the offscreen viewer can take its
        # time to exit, so escalate to SIGKILL.
        try:
            os.killpg(viewer.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            viewer.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(viewer.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    out = os.path.join(ROOT, "assets", "demo.gif")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "12",
            "-pattern_type",
            "glob",
            "-i",
            f"{frames}/frame_*.png",
            "-vf",
            f"{CROP},scale=960:-2:flags=lanczos,split[s0][s1];"
            "[s0]palettegen=max_colors=256:stats_mode=full[p];"
            "[s1][p]paletteuse=dither=sierra2_4a",
            "-loop",
            "0",
            out,
        ],
        check=True,
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
