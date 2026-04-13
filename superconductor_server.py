"""Stateful Magenta RealTime server.

Extends magenta_rt.server.Server with a request/response endpoint that
accepts an optional caller-provided generation state and returns both the
generated audio chunk *and* the resulting state. This lets an external
scheduler queue chunks, fork the queue when the recipe changes, and
regenerate audio from arbitrary points by replaying the right state.
"""

import base64
import pickle

import numpy as np
from absl import flags
from aiohttp import web

from magenta_rt import server as magenta_rt_server
from magenta_rt import system as system_lib

_PORT = flags.DEFINE_integer("sc_port", 8000, "Port to listen on.")
_TAG = flags.DEFINE_enum(
    "sc_tag", "large", ["large", "base", "mock"], "Model tag (use 'mock' for testing)."
)
_DEVICE = flags.DEFINE_string("sc_device", "gpu", "Device to use.")
_SIMULATE_LATENCY = flags.DEFINE_float(
    "sc_simulate_latency", 0.0, "Additional simulated latency in seconds."
)


def _encode_state(state) -> str:
    return base64.b64encode(pickle.dumps(state)).decode("ascii")


def _decode_state(encoded: str):
    return pickle.loads(base64.b64decode(encoded.encode("ascii")))


def _encode_audio(samples: np.ndarray) -> str:
    return base64.b64encode(samples.astype(np.float32).tobytes()).decode("ascii")


class SuperConductorServer(magenta_rt_server.Server):
    """Magenta RT server with a stateful /generate_chunk endpoint."""

    def __init__(self, system, port: int = 8000, simulate_latency: float = 0.0):
        super().__init__(
            system=system, port=port, simulate_latency=simulate_latency
        )
        self._app.router.add_post("/generate_chunk", self.handle_generate_chunk)
        self._app.router.add_post("/init_state", self.handle_init_state)

    async def handle_init_state(self, request: web.Request) -> web.Response:
        del request
        state = self._system.init_state()
        return web.json_response(
            {"state": _encode_state(state)},
            headers={"Access-Control-Allow-Origin": "*"},
        )

    async def handle_generate_chunk(self, request: web.Request) -> web.Response:
        """Generate one chunk from a caller-supplied state and style.

        Request JSON:
          {
            "style": [float, ...],         # required, length == style_embedding_dim
            "state": "<b64 pickle>",       # optional; defaults to a fresh init_state
            "generation_kwargs": { ... }   # optional
          }

        Response JSON:
          {
            "audio": "<b64 float32>",
            "state": "<b64 pickle>",       # state AFTER this chunk was generated
            "shape": [num_samples, num_channels],
            "sample_rate": int,
            "num_channels": int
          }
        """
        try:
            data = await request.json()
        except Exception as e:
            return web.json_response(
                {"error": f"invalid json: {e}"}, status=400
            )

        style = data.get("style")
        if style is None:
            return web.json_response({"error": "missing 'style'"}, status=400)
        try:
            style_emb = np.array(style, dtype=np.float32)
        except Exception as e:
            return web.json_response(
                {"error": f"bad style: {e}"}, status=400
            )
        style_dim = self._system.config.style_embedding_dim
        if style_emb.shape != (style_dim,):
            return web.json_response(
                {
                    "error": (
                        f"style shape {tuple(style_emb.shape)}, "
                        f"expected ({style_dim},)"
                    )
                },
                status=400,
            )

        encoded_state = data.get("state")
        if encoded_state is None:
            state = self._system.init_state()
        else:
            try:
                state = _decode_state(encoded_state)
            except Exception as e:  # pylint: disable=broad-exception-caught
                return web.json_response(
                    {"error": f"bad state: {e}"}, status=400
                )

        gen_kwargs = magenta_rt_server._parse_kwargs(data.get("generation_kwargs") or {})

        try:
            chunk, new_state = await self._run_in_executor(
                self._system.generate_chunk,
                state=state,
                style=style_emb,
                **gen_kwargs,
            )
        except Exception as e:
            return web.json_response(
                {"error": f"generation failed: {e}"}, status=500
            )

        samples = np.asarray(chunk.samples, dtype=np.float32)
        return web.json_response(
            {
                "audio": _encode_audio(samples),
                "state": _encode_state(new_state),
                "shape": list(samples.shape),
                "sample_rate": self._system.sample_rate,
                "num_channels": self._system.num_channels,
            },
            headers={"Access-Control-Allow-Origin": "*"},
        )


def main(_):
    if _TAG.value == "mock":
        system = system_lib.MockMagentaRT(synthesis_type="sine")
    else:
        system = system_lib.MagentaRT(
            tag=_TAG.value,
            device=_DEVICE.value,
            lazy=False,
        )
    server = SuperConductorServer(
        system=system,
        port=_PORT.value,
        simulate_latency=_SIMULATE_LATENCY.value,
    )
    server.run()


if __name__ == "__main__":
    from absl import app as absl_app

    absl_app.run(main)
