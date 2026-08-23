"""Small method-extension surface for the shared D-OPSD training loop."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence


class TeacherContextProvider(Protocol):
    def context_embeddings(
        self,
        *,
        prompts: Sequence[str],
        optimizer_step: int,
        gradient_accumulation_microstep: int,
        timestep_index: int,
        source_row_ids: Sequence[str] | None = None,
    ) -> Any: ...

    def close(self) -> None: ...


class DStepObserver(Protocol):
    """Read-only observations at the five canonical D-step seams."""

    def on_stopped_microbatch(
        self,
        *,
        step: int,
        microbatch: int,
        coordinate: Any,
        state: Any,
        tau: Any,
        student_context: Any,
        teacher_target: Any,
    ) -> None: ...

    def on_pre_clip(
        self,
        *,
        step: int,
        parameter_inventory: Sequence[str],
        g_D_raw: Mapping[str, Any],
        loss_mean: Any,
        optimizer_view: Any,
    ) -> None: ...

    def on_post_clip(
        self,
        *,
        step: int,
        parameter_inventory: Sequence[str],
        g_D_clip: Mapping[str, Any],
        clip_branch: str,
        optimizer_view: Any,
    ) -> None: ...

    def on_post_optimizer(
        self,
        *,
        step: int,
        parameter_inventory: Sequence[str],
        theta_D: Mapping[str, Any],
        delta_D_actual: Mapping[str, Any],
        optimizer_view: Any,
    ) -> None: ...

    def on_post_ema(
        self,
        *,
        step: int,
        parameter_inventory: Sequence[str],
        student_successor: Mapping[str, Any],
        ema_successor: Mapping[str, Any],
        optimizer_view: Any,
    ) -> None: ...


@dataclass(frozen=True)
class DopsdExtensionContext:
    dataset_jsonl: Path
    vl_model: Any
    processor: Any
    device: Any
    dtype: Any
    output_dir: str
    is_main_process: bool
    embedding_fn: Any


class DopsdTrainingExtension:
    """Default no-op extension used by the upstream D-OPSD CLI."""

    def teacher_conditioning_mode(self) -> str:
        """How the teacher receives its prompt embeddings.

        ``multimodal`` is the historical D-OPSD path: load Qwen3-VL and build
        image-augmented embeddings.  A method may select ``student_text`` to
        pass the exact same embedding object to teacher and student and avoid
        loading a visual-language model that it cannot use.
        """

        return "multimodal"

    def teacher_adapter_update_mode(self) -> str:
        """How the frozen adapter is updated after an optimizer step."""

        return "ema"

    def student_prompt_keys(self, *, evaluation: bool) -> Sequence[str]:
        if evaluation:
            return (
                "short_en",
                "short_zh",
                "medium_zh",
                "medium_en",
                "user_prompt_en",
                "user_prompt_zh",
            )
        return (
            "short_en",
            "detailed_en",
            "short_zh",
            "detailed_zh",
            "medium_zh",
            "medium_en",
            "user_prompt_en",
            "user_prompt_zh",
        )

    def initialize_adapter_state(self, transformer: Any) -> None:
        return None

    def prepared_adapter_state(self, transformer: Any) -> None:
        """Adjust the adapters once the runtime has finished preparing them.

        `initialize_adapter_state` runs on the model as PEFT built it, which
        is not the model that trains: DeepSpeed casts the whole module,
        frozen adapters included, when it initializes. A dtype an extension
        needs to hold across the step has to be restored here, after that
        cast, or it is silently undone.
        """

        return None

    def validate_teacher_adapter_state(
        self,
        transformer: Any,
        *,
        optimizer_step: int,
        event: str,
    ) -> None:
        """Validate a method-owned teacher at initialization/checkpoint seams."""

        return None

    def validate_student_prompts(
        self,
        prompts: Sequence[str],
        source_row_ids: Sequence[str] | None = None,
    ) -> None:
        return None

    def build_teacher_context(
        self,
        context: DopsdExtensionContext,
    ) -> TeacherContextProvider | None:
        return None

    def d_step_observer(self) -> DStepObserver | None:
        """Return a read-only canonical-D observer; default execution has none."""

        return None

    def auxiliary_loss(
        self,
        *,
        gen_model: Any,
        pipeline: Any,
        accelerator: Any,
        optimizer_step: int,
        micro_step: int,
    ) -> Any | None:
        """Optional extra loss term added to the accumulated D-OPSD loss.

        Returning ``None`` (the default) leaves the historical objective
        byte-identical.
        """

        return None

    def on_optimizer_step_end(
        self,
        *,
        gen_model: Any,
        pipeline: Any,
        accelerator: Any,
        optimizer_step: int,
    ) -> None:
        """Called once per optimizer step, after the EMA update."""

        return None


_D_STEP_OBSERVER_METHODS = (
    "on_stopped_microbatch",
    "on_pre_clip",
    "on_post_clip",
    "on_post_optimizer",
    "on_post_ema",
)


def require_d_step_observer(value: Any) -> DStepObserver | None:
    """Validate the exact return-only observer protocol without adapting it."""

    if value is None:
        return None
    missing = tuple(
        name for name in _D_STEP_OBSERVER_METHODS if not callable(getattr(value, name, None))
    )
    if missing:
        raise TypeError(
            "D-step observer omits required methods: " + ", ".join(missing)
        )
    return value


def observed_clip_branch(grad_norm: Any, max_grad_norm: float) -> str:
    """Name the canonical clipping branch from its pre-clip global norm."""

    value = float(grad_norm.detach().float().item()) if hasattr(grad_norm, "detach") else float(grad_norm)
    limit = float(max_grad_norm)
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError("D-step observer gradient norm is invalid")
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("D-step observer clipping threshold is invalid")
    return "clipped" if value > limit else "unchanged"


def _tensor_receipt(value: Any, *, bytewise: bool = True) -> tuple[Any, ...]:
    import torch

    digest = None
    if bytewise:
        byte_view = (
            value.detach().contiguous().reshape(-1).view(dtype=torch.uint8).cpu()
        )
        digest = hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()
    return (
        id(value),
        int(value.data_ptr()),
        int(value._version),
        tuple(value.shape),
        str(value.dtype),
        str(value.device),
        bool(value.requires_grad),
        digest,
    )


def _structure_receipt(value: Any, seen: set[int] | None = None) -> Any:
    """Cheap mutation receipt for observer-scoped live state."""

    import torch

    active = set() if seen is None else seen
    if isinstance(value, torch.Tensor):
        return ("tensor", _tensor_receipt(value))
    if value is None or type(value) in {str, bool, int, float, bytes}:
        return (type(value).__name__, value)
    identity = id(value)
    if identity in active:
        return ("cycle", identity)
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            rows = []
            for key, item in value.items():
                key_receipt = (
                    ("tensor-key", id(key))
                    if isinstance(key, torch.Tensor)
                    else _structure_receipt(key, active)
                )
                rows.append((repr(key_receipt), key_receipt, _structure_receipt(item, active)))
            return ("mapping", tuple((key, item) for _, key, item in sorted(rows)))
        if type(value) in {tuple, list}:
            return (
                type(value).__name__,
                tuple(_structure_receipt(item, active) for item in value),
            )
        if type(value) is set:
            return (
                "set",
                tuple(sorted(repr(_structure_receipt(item, active)) for item in value)),
            )
        return ("object", type(value).__qualname__, identity, repr(value))
    finally:
        active.remove(identity)


def _module_receipt(module: Any) -> Any:
    parameters = []
    for name, parameter in module.named_parameters():
        gradient = parameter.grad
        parameters.append(
            (
                name,
                _tensor_receipt(parameter, bytewise=True),
                (
                    None
                    if gradient is None
                    else _tensor_receipt(gradient, bytewise=True)
                ),
            )
        )
    buffers = tuple(
        (name, _tensor_receipt(buffer, bytewise=True))
        for name, buffer in module.named_buffers()
    )
    modes = []
    for name, child in module.named_modules():
        hooks = tuple(
            (
                attribute,
                tuple((key, id(callback)) for key, callback in getattr(child, attribute, {}).items()),
            )
            for attribute in (
                "_forward_pre_hooks",
                "_forward_hooks",
                "_backward_pre_hooks",
                "_backward_hooks",
            )
        )
        adapter_state = tuple(
            (attribute, repr(getattr(child, attribute)))
            for attribute in ("active_adapter", "active_adapters", "_active_adapter")
            if hasattr(child, attribute)
        )
        modes.append((name, bool(child.training), hooks, adapter_state))
    return tuple(parameters), buffers, tuple(modes)


def _optimizer_chain_receipt(optimizer: Any) -> Any:
    rows = []
    current = optimizer
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        rows.append(
            (
                type(current).__qualname__,
                _structure_receipt(getattr(current, "state", None)),
                _structure_receipt(getattr(current, "param_groups", None)),
            )
        )
        nested = getattr(current, "optimizer", None)
        current = None if nested is current else nested
    return tuple(rows)


def _rng_receipt() -> Any:
    import torch

    cpu = bytes(torch.get_rng_state().detach().cpu().tolist())
    cuda = ()
    if torch.cuda.is_available():
        cuda = tuple(
            bytes(state.detach().cpu().tolist()) for state in torch.cuda.get_rng_state_all()
        )
    numpy_state = None
    try:
        import numpy

        state = numpy.random.get_state()
        numpy_state = (state[0], state[1].tobytes(), state[2:])
    except ImportError:
        pass
    return random.getstate(), cpu, cuda, numpy_state


def _guard_receipt(
    module: Any,
    optimizer: Any,
    accelerator: Any,
    observed_arguments: Mapping[str, Any],
) -> Any:
    scaler = getattr(accelerator, "scaler", None)
    scaler_state = None if scaler is None else _structure_receipt(scaler.state_dict())
    return (
        _module_receipt(module),
        _optimizer_chain_receipt(optimizer),
        scaler_state,
        _rng_receipt(),
        _structure_receipt(observed_arguments),
    )


def _frozen_observer_view(value: Any) -> Any:
    import copy
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _frozen_observer_view(item) for key, item in value.items()}
        )
    if type(value) in {tuple, list}:
        return tuple(_frozen_observer_view(item) for item in value)
    if value is None or type(value) in {str, bool, int, float, bytes}:
        return value
    return copy.deepcopy(value)


def _invoke_d_step_observer(
    observer: DStepObserver,
    method: str,
    *,
    module: Any,
    optimizer: Any,
    accelerator: Any,
    arguments: Mapping[str, Any],
) -> None:
    """Call one observer method and reject returns or any guarded mutation."""

    if method not in _D_STEP_OBSERVER_METHODS:
        raise ValueError(f"unknown D-step observer method: {method}")
    observed = _frozen_observer_view(arguments)
    before = _guard_receipt(module, optimizer, accelerator, observed)
    result = getattr(observer, method)(**observed)
    after = _guard_receipt(module, optimizer, accelerator, observed)
    if result is not None:
        raise TypeError(f"D-step observer {method} must return None")
    if after != before:
        raise RuntimeError(f"D-step observer {method} mutated guarded state")


def _ordered_student_parameters(
    module: Any, *, require_float32: bool = True
) -> tuple[tuple[str, Any], ...]:
    """The trainable student LoRA rows, in canonical name order.

    ``require_float32`` is false on the ZeRO-2 path, where DeepSpeed casts the
    module to BF16 on initialization and keeps the FP32 master weights in its
    own partition. The observer reports that partition, not this view, so
    demanding FP32 here would refuse a configuration it is written to serve.
    """

    import torch

    rows = tuple(
        sorted(
            (
                (name, parameter)
                for name, parameter in module.named_parameters()
                if parameter.requires_grad
            ),
            key=lambda row: row[0],
        )
    )
    if not rows:
        raise RuntimeError("D-step observer found no trainable parameters")
    for name, parameter in rows:
        if ".student." not in name:
            raise RuntimeError(
                f"D-step observer trainable parameter is not student LoRA: {name}"
            )
        if require_float32 and parameter.dtype != torch.float32:
            raise RuntimeError(f"D-step observer parameter is not FP32: {name}")
    return rows


def _gradient_view(rows: Sequence[tuple[str, Any]], label: str) -> dict[str, Any]:
    import torch

    output = {}
    for name, parameter in rows:
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(f"D-step observer {label} gradient is missing: {name}")
        value = gradient.detach().to(dtype=torch.float32).clone()
        if not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"D-step observer {label} gradient is non-finite: {name}")
        output[name] = value
    return output


def _parameter_view(rows: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    return {name: parameter.detach().clone() for name, parameter in rows}


def _teacher_view(module: Any, rows: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    named = dict(module.named_parameters())
    output = {}
    for name, _ in rows:
        teacher_name = name.replace(".student.", ".teacher.")
        if teacher_name == name or teacher_name not in named:
            raise RuntimeError(
                f"D-step observer teacher coordinate is missing for {name}"
            )
        output[name] = named[teacher_name].detach().clone()
    return output


def _deepest_optimizer(optimizer: Any) -> Any:
    current = optimizer
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        nested = getattr(current, "optimizer", None)
        if nested is None or nested is current:
            return current
        current = nested
    raise RuntimeError("D-step observer optimizer wrapper cycle")


def _deepspeed_zero_optimizer(optimizer: Any) -> Any | None:
    current = optimizer
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if all(
            hasattr(current, attribute)
            for attribute in (
                "averaged_gradients",
                "bit16_groups",
                "single_partition_of_fp32_groups",
            )
        ):
            return current
        nested = getattr(current, "optimizer", None)
        current = None if nested is current else nested
    return None


def _deepspeed_gradient_view(
    optimizer: Any, rows: Sequence[tuple[str, Any]]
) -> dict[str, Any]:
    """Recover one-rank ZeRO-2 accumulated grads before its hidden step."""

    import torch

    zero = _deepspeed_zero_optimizer(optimizer)
    if zero is None:
        raise RuntimeError("D-step observer found no ZeRO optimizer")
    import torch.distributed as distributed

    if (
        not distributed.is_initialized()
        or distributed.get_world_size(group=zero.dp_process_group) != 1
        or not bool(getattr(zero, "partition_gradients", False))
        or bool(getattr(zero, "cpu_offload", False))
    ):
        raise RuntimeError(
            "D-step observer bridge requires one-rank, non-offloaded ZeRO-2"
        )
    loss_scale = float(getattr(zero, "loss_scale", 1.0))
    if loss_scale != 1.0:
        raise RuntimeError("D-step observer bridge requires unscaled BF16 ZeRO grads")
    by_parameter = {id(parameter): name for name, parameter in rows}
    gradients: dict[str, Any] = {}
    groups = tuple(getattr(zero, "bit16_groups"))
    averaged = getattr(zero, "averaged_gradients")
    for index, parameters in enumerate(groups):
        values = averaged[index]
        if not isinstance(values, (tuple, list)) or len(values) != len(parameters):
            raise RuntimeError("D-step observer ZeRO gradient group differs")
        for parameter, gradient in zip(parameters, values):
            name = by_parameter.get(id(parameter))
            if name is None:
                continue
            if gradient is None or gradient.shape != parameter.shape:
                raise RuntimeError(
                    f"D-step observer ZeRO gradient metadata differs: {name}"
                )
            value = gradient.detach().to(dtype=torch.float32).clone()
            if not bool(torch.isfinite(value).all().item()):
                raise RuntimeError(
                    f"D-step observer ZeRO gradient is non-finite: {name}"
                )
            if name in gradients:
                raise RuntimeError(
                    f"D-step observer ZeRO gradient repeats parameter: {name}"
                )
            gradients[name] = value
    inventory = tuple(name for name, _ in rows)
    if set(gradients) != set(inventory):
        raise RuntimeError("D-step observer ZeRO gradient inventory differs")
    return {name: gradients[name] for name in inventory}


def _deepspeed_parameter_view(
    optimizer: Any, rows: Sequence[tuple[str, Any]]
) -> dict[str, Any]:
    """Read theta from one-rank ZeRO-2's FP32 master partition.

    The module's own parameters are BF16 under this configuration: DeepSpeed
    casts them on initialization and updates the FP32 copies it partitions
    here. Reading the module would report a displacement quantized to BF16
    while the gradients beside it are the FP32 ones AdamW consumed.
    """

    import torch

    zero = _deepspeed_zero_optimizer(optimizer)
    if zero is None:
        raise RuntimeError("D-step observer found no ZeRO optimizer")
    by_parameter = {id(parameter): name for name, parameter in rows}
    bit16_groups = tuple(getattr(zero, "bit16_groups"))
    flat_groups = tuple(getattr(zero, "single_partition_of_fp32_groups"))
    if len(bit16_groups) != len(flat_groups):
        raise RuntimeError("D-step observer ZeRO optimizer groups differ")
    parameters_view: dict[str, Any] = {}
    for parameters, flat in zip(bit16_groups, flat_groups):
        if flat.dtype != torch.float32:
            raise RuntimeError("D-step observer ZeRO master partition is not FP32")
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            name = by_parameter.get(id(parameter))
            if name is not None:
                if flat.numel() < offset + count:
                    raise RuntimeError(
                        f"D-step observer ZeRO parameter layout differs: {name}"
                    )
                value = (
                    flat.detach()[offset : offset + count]
                    .view(parameter.shape)
                    .clone()
                )
                if not bool(torch.isfinite(value).all().item()):
                    raise RuntimeError(
                        f"D-step observer ZeRO parameter is non-finite: {name}"
                    )
                if name in parameters_view:
                    raise RuntimeError(
                        f"D-step observer ZeRO parameter repeats: {name}"
                    )
                parameters_view[name] = value
            offset += count
        if offset > flat.numel():
            raise RuntimeError("D-step observer ZeRO flat parameter is truncated")
    inventory = tuple(name for name, _ in rows)
    if set(parameters_view) != set(inventory):
        raise RuntimeError("D-step observer ZeRO parameter inventory differs")
    return {name: parameters_view[name] for name in inventory}


def _deepspeed_clipped_view(
    optimizer: Any,
    gradients: Mapping[str, Any],
    *,
    maximum: float,
) -> tuple[dict[str, Any], str]:
    """Apply the exact one-rank ZeRO-2 L2 clip coefficient to cloned grads."""

    import torch

    if not gradients:
        raise RuntimeError("D-step observer ZeRO gradient inventory is empty")
    zero = _deepspeed_zero_optimizer(optimizer)
    if zero is None:
        raise RuntimeError("D-step observer found no ZeRO optimizer")
    limit = float(maximum)
    configured_limit = float(getattr(zero, "clip_grad", float("nan")))
    if configured_limit != limit:
        raise RuntimeError("D-step observer and ZeRO clipping thresholds differ")
    total = zero.scaled_global_norm().detach().float() / float(zero.loss_scale)
    if not bool(torch.isfinite(total).item()):
        raise RuntimeError("D-step observer ZeRO gradient norm is non-finite")
    coefficient = torch.clamp((total + 1e-6) / limit, min=1.0).reciprocal()
    clipped = {
        name: (value * coefficient.to(value.device)).to(dtype=torch.float32)
        for name, value in gradients.items()
    }
    branch = "clipped" if float(total.item()) > limit else "unchanged"
    return clipped, branch


def _deepspeed_named_state(
    optimizer: Any, rows: Sequence[tuple[str, Any]]
) -> tuple[Any, dict[str, Any]] | None:
    """Unflatten one-rank ZeRO-2 Adam state into canonical model names."""

    zero = _deepspeed_zero_optimizer(optimizer)
    if zero is None:
        return None
    base = getattr(zero, "optimizer", None)
    if base is None:
        raise RuntimeError("D-step observer ZeRO optimizer lacks its base optimizer")
    by_parameter = {id(parameter): name for name, parameter in rows}
    named: dict[str, Any] = {}
    bit16_groups = tuple(getattr(zero, "bit16_groups"))
    flat_groups = tuple(getattr(zero, "single_partition_of_fp32_groups"))
    if len(bit16_groups) != len(flat_groups):
        raise RuntimeError("D-step observer ZeRO optimizer groups differ")
    for parameters, flat in zip(bit16_groups, flat_groups):
        state = getattr(base, "state", {}).get(flat)
        if state is None:
            continue
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            name = by_parameter.get(id(parameter))
            if name is not None:
                row = {}
                for field, value in state.items():
                    if field in {"exp_avg", "exp_avg_sq"}:
                        if value.ndim != 1 or value.numel() < offset + count:
                            raise RuntimeError(
                                f"D-step observer ZeRO {field} layout differs"
                            )
                        # Held at the partition's own precision: casting to
                        # the BF16 module dtype would report a coarser moment
                        # than the optimizer used.
                        row[field] = (
                            value.detach()[offset : offset + count]
                            .view(parameter.shape)
                            .clone()
                        )
                    else:
                        row[field] = _frozen_observer_view(value)
                if name in named:
                    raise RuntimeError(
                        f"D-step observer ZeRO state repeats parameter: {name}"
                    )
                named[name] = MappingProxyType(row)
            offset += count
        if offset > flat.numel():
            raise RuntimeError("D-step observer ZeRO flat parameter is truncated")
    inventory = tuple(name for name, _ in rows)
    if named and set(named) != set(inventory):
        raise RuntimeError("D-step observer ZeRO optimizer-state inventory differs")
    return base, {name: named[name] for name in inventory if name in named}


def _optimizer_view(
    optimizer: Any,
    rows: Sequence[tuple[str, Any]],
    accelerator: Any,
) -> Mapping[str, Any]:
    deepspeed_state = _deepspeed_named_state(optimizer, rows)
    if deepspeed_state is None:
        base = _deepest_optimizer(optimizer)
        state = getattr(base, "state", {})
        by_parameter = {id(parameter): name for name, parameter in rows}
        named_state = {}
        for parameter, values in state.items():
            name = by_parameter.get(id(parameter))
            if name is not None:
                named_state[name] = _frozen_observer_view(values)
        named_state = {name: named_state[name] for name in sorted(named_state)}
    else:
        base, named_state = deepspeed_state
        by_parameter = {id(parameter): name for name, parameter in rows}
    groups = []
    for group in getattr(base, "param_groups", ()):
        names = tuple(
            by_parameter[id(parameter)]
            for parameter in group.get("params", ())
            if id(parameter) in by_parameter
        )
        values = {
            key: _frozen_observer_view(value)
            for key, value in group.items()
            if key != "params"
        }
        values["parameter_names"] = names
        groups.append(MappingProxyType(values))
    scaler = getattr(accelerator, "scaler", None)
    return MappingProxyType(
        {
            "optimizer_class": type(base).__qualname__,
            "state": MappingProxyType(named_state),
            "param_groups": tuple(groups),
            "scaler_state": (
                None
                if scaler is None
                else _frozen_observer_view(scaler.state_dict())
            ),
        }
    )


class DStepObserverSession:
    """One enabled observer's state; absent entirely on the default path."""

    def __init__(
        self,
        observer: DStepObserver,
        *,
        module: Any,
        optimizer: Any,
        accelerator: Any,
        gradient_accumulation_steps: int,
        max_grad_norm: float | None = None,
    ) -> None:
        self._observer = require_d_step_observer(observer)
        if self._observer is None:
            raise TypeError("D-step observer session requires an observer")
        if (
            type(gradient_accumulation_steps) is not int
            or gradient_accumulation_steps <= 0
        ):
            raise ValueError("D-step observer accumulation count is invalid")
        self._module = module
        self._optimizer = optimizer
        self._accelerator = accelerator
        unwrapped = accelerator.unwrap_model(module)
        if unwrapped is None or not hasattr(unwrapped, "named_parameters"):
            raise RuntimeError("D-step observer could not unwrap the model")
        self._parameter_module = unwrapped
        engine_wrapper = getattr(accelerator, "deepspeed_engine_wrapped", None)
        self._rows = _ordered_student_parameters(
            self._parameter_module, require_float32=engine_wrapper is None
        )
        self._inventory = tuple(name for name, _ in self._rows)
        self._expected_microbatches = gradient_accumulation_steps
        self._step: int | None = None
        self._losses: list[Any] = []
        self._theta_t: dict[str, Any] | None = None
        self._hidden_engine = None
        self._max_grad_norm = max_grad_norm
        if engine_wrapper is not None:
            engine = getattr(engine_wrapper, "engine", None)
            if engine is None or _deepspeed_zero_optimizer(optimizer) is None:
                raise RuntimeError(
                    "D-step observer DeepSpeed optimizer boundary differs"
                )
            if (
                max_grad_norm is None
                or not math.isfinite(float(max_grad_norm))
                or float(max_grad_norm) <= 0.0
            ):
                raise ValueError(
                    "D-step observer DeepSpeed clipping threshold is invalid"
                )
            marker = "_dopsd_observer_original_step"
            if hasattr(engine, marker):
                raise RuntimeError("D-step observer engine step is already wrapped")
            original_step = engine.step

            def observed_engine_step(*args: Any, **kwargs: Any) -> Any:
                raw = _deepspeed_gradient_view(self._optimizer, self._rows)
                clipped, branch = _deepspeed_clipped_view(
                    self._optimizer,
                    raw,
                    maximum=float(self._max_grad_norm),
                )
                self.on_pre_clip(step=int(self._step), raw_gradients=raw)
                self.on_post_clip(
                    step=int(self._step),
                    clip_branch=branch,
                    clipped_gradients=clipped,
                )
                result = original_step(*args, **kwargs)
                self.on_post_optimizer(step=int(self._step))
                return result

            setattr(engine, marker, original_step)
            engine.step = observed_engine_step
            self._hidden_engine = engine

    def _student_view(self) -> dict[str, Any]:
        """Theta, from whichever copy the optimizer actually updates."""

        if self._hidden_engine is None:
            return _parameter_view(self._rows)
        return _deepspeed_parameter_view(self._optimizer, self._rows)

    @property
    def parameter_inventory(self) -> tuple[str, ...]:
        return self._inventory

    @property
    def owns_optimizer_step_hooks(self) -> bool:
        """Whether DeepSpeed executes observer seams inside ``backward``."""

        return self._hidden_engine is not None

    def _call(self, method: str, **arguments: Any) -> None:
        _invoke_d_step_observer(
            self._observer,
            method,
            module=self._module,
            optimizer=self._optimizer,
            accelerator=self._accelerator,
            arguments=arguments,
        )

    def on_stopped_microbatch(
        self,
        *,
        step: int,
        microbatch: int,
        coordinate: Any,
        state: Any,
        tau: Any,
        student_context: Any,
        teacher_target: Any,
    ) -> None:
        self._call(
            "on_stopped_microbatch",
            step=step,
            microbatch=microbatch,
            coordinate=coordinate,
            state=state,
            tau=tau,
            student_context=student_context,
            teacher_target=teacher_target,
        )

    def record_microbatch_loss(self, *, step: int, microbatch: int, loss: Any) -> None:
        import torch

        if self._step is None:
            self._step = step
        if step != self._step or microbatch != len(self._losses):
            raise RuntimeError("D-step observer microbatch order differs")
        value = loss.detach().to(dtype=torch.float64).clone()
        if value.ndim != 0 or not bool(torch.isfinite(value).item()):
            raise RuntimeError("D-step observer microbatch loss is invalid")
        self._losses.append(value)

    def on_pre_clip(
        self,
        *,
        step: int,
        raw_gradients: Mapping[str, Any] | None = None,
    ) -> None:
        if step != self._step or len(self._losses) != self._expected_microbatches:
            raise RuntimeError("D-step observer pre-clip accumulation is incomplete")
        # Summed in FP64, not in the losses' own dtype: under BF16 autocast
        # each microbatch loss is BF16, and adding four of them there rounds
        # the mean to BF16 before anything downstream sees it.
        total = self._losses[0].double()
        for loss in self._losses[1:]:
            total = total + loss.double()
        self._theta_t = self._student_view()
        self._call(
            "on_pre_clip",
            step=step,
            parameter_inventory=self._inventory,
            g_D_raw=(
                _gradient_view(self._rows, "raw D")
                if raw_gradients is None
                else raw_gradients
            ),
            loss_mean=total / float(self._expected_microbatches),
            optimizer_view=_optimizer_view(
                self._optimizer, self._rows, self._accelerator
            ),
        )

    def on_post_clip(
        self,
        *,
        step: int,
        clip_branch: str,
        clipped_gradients: Mapping[str, Any] | None = None,
    ) -> None:
        if step != self._step or self._theta_t is None:
            raise RuntimeError("D-step observer post-clip order differs")
        if type(clip_branch) is not str or not clip_branch:
            raise ValueError("D-step observer clip branch is empty")
        self._call(
            "on_post_clip",
            step=step,
            parameter_inventory=self._inventory,
            g_D_clip=(
                _gradient_view(self._rows, "clipped D")
                if clipped_gradients is None
                else clipped_gradients
            ),
            clip_branch=clip_branch,
            optimizer_view=_optimizer_view(
                self._optimizer, self._rows, self._accelerator
            ),
        )

    def on_post_optimizer(self, *, step: int) -> None:
        import torch

        if step != self._step or self._theta_t is None:
            raise RuntimeError("D-step observer post-optimizer order differs")
        theta_d = self._student_view()
        delta = {
            name: theta_d[name].to(dtype=torch.float64)
            - self._theta_t[name].to(dtype=torch.float64)
            for name in self._inventory
        }
        self._call(
            "on_post_optimizer",
            step=step,
            parameter_inventory=self._inventory,
            theta_D=theta_d,
            delta_D_actual=delta,
            optimizer_view=_optimizer_view(
                self._optimizer, self._rows, self._accelerator
            ),
        )

    def on_post_ema(self, *, step: int) -> None:
        if step != self._step or self._theta_t is None:
            raise RuntimeError("D-step observer post-EMA order differs")
        self._call(
            "on_post_ema",
            step=step,
            parameter_inventory=self._inventory,
            student_successor=self._student_view(),
            ema_successor=_teacher_view(self._parameter_module, self._rows),
            optimizer_view=_optimizer_view(
                self._optimizer, self._rows, self._accelerator
            ),
        )
        self._step = None
        self._losses.clear()
        self._theta_t = None


__all__ = [
    "DStepObserver",
    "DStepObserverSession",
    "DopsdExtensionContext",
    "DopsdTrainingExtension",
    "TeacherContextProvider",
    "observed_clip_branch",
    "require_d_step_observer",
]
