.. SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

.. _concept_foundation_models:

Foundation Models
=================

Most OpenSTEF models learn from a single target: you fit an XGBoost or LightGBM
forecaster on that meter's history, and it predicts that meter. A foundation model
inverts this. It is pretrained on a large corpus of time series, so it forecasts a series
it has never seen without any per-target training.

``openstef-foundation-models`` is the home for these models in OpenSTEF. It provides the
model-agnostic machinery they share (an ONNX inference backend, host-aware provider
selection, a checkpoint catalog, and the workflow and backtesting adapters) plus the
model families that run on top of it. Each family is wrapped behind the same
:class:`~openstef_models.models.forecasting.forecaster.Forecaster` interface the rest of
OpenSTEF uses. Chronos-2 is the first family the package ships; later families reuse the
same backend, workflow, and backtesting surface, so the examples below carry over.

.. note::

   For the trainable models this sits alongside, see :doc:`models`. For evaluating a
   foundation model against those models, see :doc:`beam`. For a runnable example, see
   :doc:`/user_guide/guides/foundation_model_forecasting_tutorial`.

When to reach for a foundation model
------------------------------------

A foundation model trades per-target tuning for zero setup. That trade is worth making in
a few concrete situations:

- **Cold start.** A new meter has no history to train on, but a foundation model can still
  forecast it from whatever context window exists.
- **Many targets, no training budget.** Fitting and storing one model per target across
  thousands of meters is expensive. One shared checkpoint forecasts all of them.
- **A zero-shot baseline.** Before investing in feature engineering and hyperparameter
  search, you want a reference number that needs no fitting.

A per-target XGBoost or LightGBM model still wins when you have ample history for a target
and can afford to tune it, because it can learn that target's specific calendar and weather
response. The two approaches coexist; :doc:`beam` is how you decide which one to ship.

Zero-shot forecasting
---------------------

A foundation model is pretrained, so there is nothing to fit. The first family in the
package, Chronos-2, is wrapped by :class:`~openstef_foundation_models.models.forecasting.chronos2_forecaster.Chronos2Forecaster`,
which reports :attr:`is_fitted` as ``True`` the moment a backend is attached, and whose
:meth:`~openstef_foundation_models.models.forecasting.chronos2_forecaster.Chronos2Forecaster.fit`
is a no-op. You hand it a window of recent target values and it returns a probabilistic
forecast directly.

The model also accepts known-future covariates. Every non-target feature column you keep is
forwarded to Chronos-2 as a covariate: its history feeds an extra context row and its values
across the horizon feed the model as known future input. Because the target and its
covariates share one attention group, the forecast can react to a covariate you already know,
such as a temperature forecast for the days ahead. Chronos-2 normalises each series itself, so
values are passed through unscaled.

The checkpoint catalog
----------------------

Each model family has a catalog of published checkpoints on the HuggingFace Hub. For
Chronos-2, the :class:`~openstef_foundation_models.models.catalog.Chronos2` catalog turns a
model size into a checkpoint reference instead of a hand-written repo id and filename:

.. code-block:: python

   from openstef_foundation_models.models import Chronos2, CheckpointVariant

   # Base model, dynamic shapes: runs on any provider.
   ref = Chronos2.BASE.checkpoint()

   # Smaller model, static shapes: the macOS CoreML path.
   ref = Chronos2.SMALL.checkpoint(CheckpointVariant.STATIC)

   # Let the host pick: static on macOS, dynamic elsewhere.
   ref = Chronos2.BASE.checkpoint(CheckpointVariant.recommended())

Two axes describe a checkpoint:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Size
     - Description
   * - ``Chronos2.BASE``
     - The full ``chronos-2`` model. The default in :class:`~openstef_foundation_models.presets.forecasting_workflow.ForecastingWorkflowConfig`.
   * - ``Chronos2.SMALL``
     - The compact ``chronos-2-small`` model. Lighter to download and run.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Variant
     - Description
   * - ``CheckpointVariant.DYNAMIC``
     - Variable input shapes. Portable across every execution provider.
   * - ``CheckpointVariant.STATIC``
     - Frozen input shapes. Required for the CoreML provider on macOS.

Each checkpoint resolves to two files: the ``.onnx`` weights and a ``.metadata.json``
sidecar. The sidecar is a :class:`~openstef_foundation_models.models.checkpoint.CheckpointMetadata`
record carrying the input names, native quantile grid, context length, horizon, resolution,
``precision`` (``fp32``, ``fp16``, or ``int8``), and a ``static_shapes`` flag. The forecaster
and the provider policy both read this metadata rather than guessing from the file. A
checkpoint already on disk is described by :class:`~openstef_foundation_models.models.checkpoint.LocalCheckpoint`
instead of downloading from the Hub.

The ONNX inference backend
--------------------------

The forecaster does not call ONNX Runtime directly. It talks to an
:class:`~openstef_foundation_models.inference.backend.InferenceBackend`, a small protocol with
a ``metadata`` property, a ``run`` method, and ``close``. The one concrete implementation today
is :class:`~openstef_foundation_models.inference.onnx_backend.OnnxBackend`, which wraps a single
ONNX Runtime session. Keeping the backend behind a protocol means a non-ONNX backend can be
added later without changing the forecaster.

A backend is built once and reused for every prediction. Loading the ONNX session is the
expensive step, so the workflow and the backtesting adapter both hold one session for their
whole lifetime rather than rebuilding it per window.

Execution provider selection
----------------------------

ONNX Runtime runs a graph through an *execution provider*: plain CPU, CUDA, TensorRT, or
CoreML on macOS. Picking the right one depends on the host and on the checkpoint, so
``openstef-foundation-models`` decides it from data rather than hardcoding.

:class:`~openstef_foundation_models.inference.provider_selection.HostCapabilities` captures the
two host facts that matter: the platform name and the set of providers ONNX Runtime reports as
available. :class:`~openstef_foundation_models.inference.provider_selection.DefaultProviderPolicy`
reads those facts plus the checkpoint metadata and returns an ordered fallback chain.

.. mermaid:: /diagrams/user_guide/concepts/foundation_models_diagram_2.mmd

The rules encode findings measured during checkpoint benchmarking:

- An ``int8`` checkpoint uses CUDA then CPU when CUDA is present, and CPU otherwise. CoreML
  cannot accelerate int8 QDQ graphs, so it is skipped for that precision.
- On macOS with CoreML available and a ``static_shapes`` checkpoint, the chain is CoreML
  (with compute units set to CPU and GPU, not the Neural Engine) then CPU. CoreML cannot run a
  dynamic-shape graph, which is why the static variant exists.
- With CUDA available, the chain is CUDA then CPU.
- Otherwise the chain is CPU alone.

TensorRT is never selected by default; it is opt-in because building its engines is slow on
first run. The strictness of provider handling depends on how you ask. If you pass an explicit
provider list, a missing accelerator raises an error, because you asked for it by name. If you
let the policy choose, only a full fall-through to CPU emits a warning.

From config to a running workflow
---------------------------------

:func:`~openstef_foundation_models.presets.forecasting_workflow.create_forecasting_workflow`
is the one-call path from a config to something you can predict with. You declare the model,
the checkpoint, the quantiles and horizons, the target column, and which features to keep:

.. code-block:: python

   from openstef_core.types import LeadTime, Q
   from openstef_foundation_models.models import Chronos2
   from openstef_foundation_models.presets.forecasting_workflow import (
       ForecastingWorkflowConfig,
       create_forecasting_workflow,
   )

   workflow = create_forecasting_workflow(
       ForecastingWorkflowConfig(
           checkpoint=Chronos2.SMALL.checkpoint(),
           quantiles=[Q(0.3), Q(0.5), Q(0.7)],
           horizons=[LeadTime.from_string("P7D")],
           target_column="load",
       )
   )

   forecast = workflow.predict(window, forecast_start=forecast_start)

The factory resolves the checkpoint, builds the ONNX session once, and wraps the
:class:`~openstef_foundation_models.models.forecasting.chronos2_forecaster.Chronos2Forecaster`
in a :class:`~openstef_models.workflows.CustomForecastingWorkflow`. A ``Selector`` in front of
the forecaster keeps the target and the covariates you listed; a ``QuantileSorter`` behind it
keeps the output quantiles monotone. The result is an ordinary OpenSTEF workflow, so the same
object that predicts here can be evaluated in :doc:`beam`.

.. mermaid:: /diagrams/user_guide/concepts/foundation_models_diagram_1.mmd

On the way out, the forecaster post-processes the raw model output: it slices the model's
fixed horizon down to the length you requested and resamples Chronos-2's native quantile grid
onto your requested quantiles.

Batching
--------

A foundation model runs a whole stack of series in one backend call. Both the forecaster
and the workflow expose a batched path next to the single-series one:

- :meth:`~openstef_foundation_models.models.forecasting.chronos2_forecaster.Chronos2Forecaster.predict`
  forecasts one input; it is a batch-of-one wrapper around the batched path.
- :meth:`~openstef_foundation_models.models.forecasting.chronos2_forecaster.Chronos2Forecaster.predict_batch`
  takes a list of inputs, concatenates them into the model's tensors, runs the ONNX session a
  single time, and returns one forecast per input in the same order.

Batching changes throughput, not the result: the forecast for a window is identical whether you
call ``predict`` in a loop or pass the windows together to ``predict_batch``. Internally every
series becomes a block of rows (the target plus one row per covariate), all the blocks are
stacked along the batch axis, the session runs once, and each series' target row is sliced back
out of the combined output. The win is one session call instead of N, which matters when you
forecast many meters or many origins at once.

Backtesting integration
------------------------

:class:`~openstef_foundation_models.integrations.beam.FoundationModelBacktestForecaster` adapts
a workflow to the openstef-beam backtesting interface. It wraps one already-built workflow and
reuses it for every backtest window, so the ONNX session loads once for the whole run rather
than once per window. Build it from a workflow with
:meth:`~openstef_foundation_models.integrations.beam.FoundationModelBacktestForecaster.from_workflow`,
which sizes the prediction window to the workflow's own horizon:

.. code-block:: python

   from openstef_foundation_models.integrations.beam import FoundationModelBacktestForecaster

   adapter = FoundationModelBacktestForecaster.from_workflow(workflow)

Because Chronos-2 is zero-shot, the adapter's ``fit`` is a no-op and the default window config
disables training. Setting ``batch_size`` above one routes consecutive backtest windows through
the batched path: beam's pipeline stacks that many windows into a single ``predict_batch`` call.
A window with no observed target history before its horizon is dropped and mapped back to a
``None`` result at its original position, so the output stays aligned with the input order.

.. seealso::

   - :ref:`concept_models` for the trainable forecasters this sits alongside.
   - :ref:`concept_beam` for backtesting a foundation model against those forecasters.
   - :doc:`/user_guide/guides/foundation_model_forecasting_tutorial` for a runnable walkthrough.
   - :doc:`/api/foundation_models` for the full ``openstef-foundation-models`` API reference.
