"""Text-to-SVG icons for parameter prompts.

The model is SVGen's Qwen2.5-3B (gitcat-404/SVGen-Qwen2.5-3B-Instruct): a
Qwen2.5-3B-Instruct fine-tune trained to write SVG. The checkpoint is a plain
Qwen2 model, so the same weights load with mlx-lm on macOS and with
transformers on the cluster. This does not use Ollama.

The weights load on the first request (a few GB). The client image bakes
them into /opt/models/svgen. On a laptop, `make setup` downloads them
(`make pull-svg` does the same). Compute nodes often have no internet.
"""
import os
import re
import sys
import threading

SVG_MODEL = "gitcat-404/SVGen-Qwen2.5-3B-Instruct"
MAX_TOKENS = 1200

# SVGen was trained as an instruct model. A short system line keeps the
# answer to one SVG element; greedy decoding keeps the markup well-formed.
SYSTEM = "Output valid SVG code only."

_SVG = re.compile(r"<svg\b[^>]*>.*?</svg>", re.IGNORECASE | re.DOTALL)
_BLOCK = re.compile(r"<(script|foreignObject|iframe|embed|object)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"</?(script|foreignObject|iframe|embed|object)\b[^>]*?/?>", re.IGNORECASE)
_HANDLER = re.compile(r"""\s+on[a-z]+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)""", re.IGNORECASE)
_JS_URL = re.compile(r"""\s(?:href|xlink:href)\s*=\s*(['"])\s*javascript:[^'"]*\1""", re.IGNORECASE)


class SvgError(RuntimeError):
    def __init__(self, message, status=503):
        super().__init__(message)
        self.status = status


def default_backend():
    """mlx on macOS, transformers on the cluster. SVG_BACKEND overrides either."""
    choice = os.environ.get("SVG_BACKEND", "").strip().lower()
    if choice in ("mlx", "transformers"):
        return choice
    return "mlx" if sys.platform == "darwin" else "transformers"


def inference_dtype(cuda, capability, mps):
    """bfloat16 on Ampere and newer, including the L40S. float16 on older CUDA GPUs and on Apple MPS."""
    if cuda and capability >= 8:
        return "bfloat16"
    if cuda or mps:
        return "float16"
    return "float32"


def chat_messages(prompt):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt.strip()},
    ]


def extract_svg(text):
    """The first <svg> element in a model reply, or None. Scripts and event
    handlers are dropped; the page only shows it as an image."""
    if not text:
        return None
    text = text.replace("```svg", "").replace("```xml", "").replace("```", "")
    match = _SVG.search(text)
    if not match:
        return None
    svg = _BLOCK.sub("", match.group(0))
    svg = _TAG.sub("", svg)
    svg = _HANDLER.sub("", svg)
    svg = _JS_URL.sub("", svg)
    if not svg.lower().lstrip().startswith("<svg"):
        return None
    if "xmlns=" not in svg[:300].lower():
        svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    return svg.strip()


class SvgModel:
    def __init__(self, model_id=SVG_MODEL, backend=None):
        self.model_id = model_id
        self.backend = backend or default_backend()
        self.ready = False
        self.error = None
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def installed(self):
        """whether this machine can load the backend, without downloading weights"""
        try:
            if self.backend == "mlx":
                import mlx_lm  # noqa: F401
            else:
                import torch  # noqa: F401
                import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def generate(self, prompt):
        """One SVG for `prompt`. Loads the weights on the first call."""
        prompt = (prompt or "").strip()
        if not prompt:
            raise SvgError("missing prompt", 400)
        with self._lock:
            self._ensure()
            text = self._complete(chat_messages(prompt[:400]))
        svg = extract_svg(text)
        if not svg:
            raise SvgError("the model did not return an SVG", 502)
        self.error = None
        return svg

    def _ensure(self):
        if self._model is not None:
            return
        try:
            if self.backend == "mlx":
                self._load_mlx()
            else:
                self._load_transformers()
        except SvgError:
            raise
        except Exception as e:
            self.ready = False
            self._model = None
            self.error = str(e)
            raise SvgError(f"could not load {self.model_id} ({self.backend}): {e}", 503) from e
        self.ready = True
        self.error = None

    def _local_files_only(self):
        """A directory baked into the image (or HF_HUB_OFFLINE) must not try the network."""
        return os.path.isdir(self.model_id) or os.environ.get("HF_HUB_OFFLINE") == "1"

    def _load_mlx(self):
        try:
            from mlx_lm import load
        except ImportError as e:
            raise SvgError("mlx-lm is not installed (pip install mlx-lm)", 503) from e
        self._model, self._tokenizer = load(self.model_id)

    def _load_transformers(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise SvgError("transformers and torch are not installed (pip install transformers)", 503) from e
        from superconductor_client.torch_device import DESCRIBE_NEEDS_GPU, cuda_state
        if cuda_state() != "ok":
            raise SvgError(DESCRIBE_NEEDS_GPU, 503)
        capability = torch.cuda.get_device_capability()[0]
        kind = inference_dtype(True, capability, False)
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[kind]
        local = self._local_files_only()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=local)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype, local_files_only=local)
        model.config.use_cache = True
        model.to("cuda")
        model.eval()
        self._model = model
        self._device = "cuda"

    def _prompt_text(self, messages):
        tok = self._tokenizer
        apply = getattr(tok, "apply_chat_template", None)
        if apply is not None:
            try:
                return apply(messages, add_generation_prompt=True, tokenize=False)
            except TypeError:
                return apply(messages, add_generation_prompt=True)
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def _complete(self, messages):
        prompt = self._prompt_text(messages)
        if self.backend == "mlx":
            return self._mlx_generate(prompt)
        return self._transformers_generate(prompt)

    def _mlx_generate(self, prompt):
        import inspect

        from mlx_lm import generate
        params = inspect.signature(generate).parameters
        kwargs = {}
        if "max_tokens" in params:
            kwargs["max_tokens"] = MAX_TOKENS
        if "verbose" in params:
            kwargs["verbose"] = False
        if "temp" in params:
            kwargs["temp"] = 0
        if "prompt" in params:
            kwargs["prompt"] = prompt
            return generate(self._model, self._tokenizer, **kwargs)
        return generate(self._model, self._tokenizer, prompt, **kwargs)

    def _transformers_generate(self, prompt):
        import torch
        tok = self._tokenizer
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=MAX_TOKENS, do_sample=False,
                pad_token_id=tok.eos_token_id)
        new = out[0, inputs["input_ids"].shape[-1]:]
        return tok.decode(new, skip_special_tokens=True)
