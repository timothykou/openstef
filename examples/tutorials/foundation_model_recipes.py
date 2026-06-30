# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0


# %% [markdown]
# # Foundation Model Recipes
#
# Task-focused snippets for `openstef-foundation-models`. The recipes use Chronos-2, the
# first model family in the package; the install, provider, batching, and backtesting steps
# carry over to later families. For the end-to-end walkthrough, see the
# {doc}`Foundation-Model Forecasting tutorial </user_guide/guides/foundation_model_forecasting_tutorial>`.
# For the design behind these pieces, see the
# {ref}`Foundation Models concept page <concept_foundation_models>`.
#
# The checkpoint, provider, and adapter cells below run during the docs build, so the output
# you see is the live result of the current API. The workflow, batching, and backtesting cells
# download and run a model, so they are shown as reference and skipped during the build.

# %% tags=["remove-cell"]
import warnings

warnings.filterwarnings("ignore")

from openstef_core.testing import setup_notebook_logging

logger = setup_notebook_logging(
    __name__,
    suppress=("huggingface_hub", "fsspec", "filelock", "openstef_core.datasets"),
)


# %% [markdown]
# ## Install one ONNX runtime
#
# `[cpu]` and `[gpu]` are mutually exclusive: `onnxruntime` and `onnxruntime-gpu` collide in
# the same environment. Pick one. uv enforces the choice through conflicting extras; pip does
# not, so choose it yourself.
#
# ```bash
# # CPU (what the meta-package installs by default)
# pip install "openstef-foundation-models[cpu]"
#
# # CUDA GPU: ships onnxruntime-gpu plus the pinned NVIDIA CUDA 12 / cuDNN 9 wheels,
# # so no system CUDA install is required.
# pip install "openstef-foundation-models[gpu]"
# ```
#
# Through the meta-package, `openstef[foundation-models]` installs the CPU runtime. For the GPU
# runtime, install `openstef-foundation-models[gpu]` directly, or in the uv workspace use the
# `dev-gpu` group (`uv sync --no-default-groups --group dev-gpu`).

# %% [markdown]
# ## Pick a checkpoint
#
# Use the `Chronos2` catalog instead of writing repo ids by hand. Each entry resolves to a
# published checkpoint reference; printing it shows the repo id and file the workflow will pull.
# A `ForecastingWorkflowConfig` with no checkpoint defaults to the base, dynamic-shape build
# shown first below.

# %%
from openstef_foundation_models.models import CheckpointVariant, Chronos2

catalog = {
    # Default: base model, dynamic shapes.
    "default (base, dynamic)": Chronos2.BASE.checkpoint(),
    # Smaller model, faster to download and run.
    "small": Chronos2.SMALL.checkpoint(),
    # Static shapes, which the default policy needs to pick CoreML on macOS.
    "base, static": Chronos2.BASE.checkpoint(CheckpointVariant.STATIC),
    # Let the host choose: static on macOS, dynamic elsewhere.
    "base, recommended": Chronos2.BASE.checkpoint(CheckpointVariant.recommended()),
}
for label, ref in catalog.items():
    print(f"{label:24} -> {ref.repo_id} :: {ref.filename}")


# %% [markdown]
# ## Run a checkpoint already on disk
#
# A file you exported or downloaded yourself is described by `LocalCheckpoint`. It needs the
# `.onnx` weights and the matching `.metadata.json` sidecar. Building the config does not read
# the files, so this validates the wiring without a model present.

# %%
from pathlib import Path

from openstef_foundation_models.models.checkpoint import LocalCheckpoint
from openstef_foundation_models.presets import ForecastingWorkflowConfig

local_config = ForecastingWorkflowConfig(
    checkpoint=LocalCheckpoint(
        path=Path("artifacts/chronos-2.onnx"),
        metadata_path=Path("artifacts/chronos-2.metadata.json"),
    ),
)
print(local_config.checkpoint)


# %% [markdown]
# ## Force a specific execution provider
#
# By default the host-aware policy chooses the provider chain. To pin it, pass an explicit
# `providers` list on the backend config. An explicit list is strict: a missing accelerator
# raises instead of silently falling back. The available provider configs are `CpuProvider`,
# `CudaProvider`, `TensorRTProvider`, and `CoreMLProvider`. TensorRT is never selected by the
# default policy, so name it explicitly if you want it.

# %%
from openstef_foundation_models.inference import CpuProvider, CudaProvider, ExecutionProvider
from openstef_foundation_models.presets import OnnxBackendConfig

providers: list[ExecutionProvider] = [CudaProvider(device_id=0), CpuProvider()]
gpu_config = ForecastingWorkflowConfig(backend=OnnxBackendConfig(providers=providers))
ordered_providers = [provider.to_ort()[0] for provider in providers]
print(f"ONNX Runtime will try these providers in order: {ordered_providers}")
print(f"Pinned on the workflow config: {gpu_config.backend.providers is providers}")


# %% [markdown]
# ## Forecast many windows in one call
#
# Pass a list of windows and their forecast starts to `predict_batch` to run the ONNX session
# once instead of once per window. The numbers match a serial loop; only throughput changes.
# This cell downloads and runs Chronos-2, so it is reference-only and skipped during the build.

# %% tags=["skip-execution"]
from datetime import datetime, timedelta

from openstef_core.testing import load_liander_dataset
from openstef_foundation_models.presets import create_forecasting_workflow
from openstef_models.utils.feature_selection import Include

workflow = create_forecasting_workflow(
    ForecastingWorkflowConfig(
        checkpoint=Chronos2.SMALL.checkpoint(),
        selected_features=Include("load"),
    ),
)
dataset = load_liander_dataset()
forecast_starts = [
    datetime.fromisoformat("2024-09-15T00:00:00+00:00"),
    datetime.fromisoformat("2024-09-29T00:00:00+00:00"),
]
windows = [
    dataset.filter_by_range(start=start - timedelta(days=60), end=start + workflow.model.max_horizon.value)
    for start in forecast_starts
]
forecasts = workflow.predict_batch(windows, forecast_start=forecast_starts)
print(f"{len(forecasts)} forecasts from a single batched backend call")


# %% [markdown]
# ## Backtest with openstef-beam
#
# Wrap the workflow in `FoundationModelBacktestForecaster`. It loads the ONNX session once and
# reuses it across every window. Set `batch_size` above one to let beam stack consecutive
# windows into a single batched call. Plug the adapter into a `BacktestPipeline` the same way
# as any other backtest forecaster; see the {ref}`Backtesting concept page <concept_beam>` for
# the pipeline.

# %% tags=["skip-execution"]
from openstef_foundation_models.integrations.beam import FoundationModelBacktestForecaster

adapter = FoundationModelBacktestForecaster.from_workflow(workflow, batch_size=16)
print(f"Adapter forecasts quantiles {adapter.quantiles} with batch size {adapter.batch_size}")
