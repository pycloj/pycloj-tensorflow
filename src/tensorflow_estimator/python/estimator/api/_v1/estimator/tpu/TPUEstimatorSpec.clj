(ns tensorflow-estimator.python.estimator.api.-v1.estimator.tpu.TPUEstimatorSpec
  "Ops and objects returned from a `model_fn` and passed to `TPUEstimator`.

  See `EstimatorSpec` for `mode`, `predictions`, `loss`, `train_op`, and
  `export_outputs`.

  For evaluation, `eval_metrics `is a tuple of `metric_fn` and `tensors`, where
  `metric_fn` runs on CPU to generate metrics and `tensors` represents the
  `Tensor`s transferred from TPU system to CPU host and passed to `metric_fn`.
  To be precise, TPU evaluation expects a slightly different signature from the
  `tf.estimator.Estimator`. While `EstimatorSpec.eval_metric_ops` expects a
  dict, `TPUEstimatorSpec.eval_metrics` is a tuple of `metric_fn` and `tensors`.
  The `tensors` could be a list of `Tensor`s or dict of names to `Tensor`s. The
  `tensors` usually specify the model logits, which are transferred back from
  TPU system to CPU host. All tensors must have be batch-major, i.e., the batch
  size is the first dimension. Once all tensors are available at CPU host from
  all shards, they are concatenated (on CPU) and passed as positional arguments
  to the `metric_fn` if `tensors` is list or keyword arguments if `tensors` is
  a dict. `metric_fn` takes the `tensors` and returns a dict from metric string
  name to the result of calling a metric function, namely a `(metric_tensor,
  update_op)` tuple. See `TPUEstimator` for MNIST example how to specify the
  `eval_metrics`.

  `scaffold_fn` is a function running on CPU to generate the `Scaffold`. This
  function should not capture any Tensors in `model_fn`.

  `host_call` is a tuple of a `function` and a list or dictionary of `tensors`
  to pass to that function and returns a list of Tensors. `host_call` currently
  works for train() and evaluate(). The Tensors returned by the function is
  executed on the CPU on every step, so there is communication overhead when
  sending tensors from TPU to CPU. To reduce the overhead, try reducing the
  size of the tensors. The `tensors` are concatenated along their major (batch)
  dimension, and so must be >= rank 1. The `host_call` is useful for writing
  summaries with `tf.contrib.summary.create_file_writer`.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tpu (import-module "tensorflow_estimator.python.estimator.api._v1.estimator.tpu"))

(defn TPUEstimatorSpec 
  "Ops and objects returned from a `model_fn` and passed to `TPUEstimator`.

  See `EstimatorSpec` for `mode`, `predictions`, `loss`, `train_op`, and
  `export_outputs`.

  For evaluation, `eval_metrics `is a tuple of `metric_fn` and `tensors`, where
  `metric_fn` runs on CPU to generate metrics and `tensors` represents the
  `Tensor`s transferred from TPU system to CPU host and passed to `metric_fn`.
  To be precise, TPU evaluation expects a slightly different signature from the
  `tf.estimator.Estimator`. While `EstimatorSpec.eval_metric_ops` expects a
  dict, `TPUEstimatorSpec.eval_metrics` is a tuple of `metric_fn` and `tensors`.
  The `tensors` could be a list of `Tensor`s or dict of names to `Tensor`s. The
  `tensors` usually specify the model logits, which are transferred back from
  TPU system to CPU host. All tensors must have be batch-major, i.e., the batch
  size is the first dimension. Once all tensors are available at CPU host from
  all shards, they are concatenated (on CPU) and passed as positional arguments
  to the `metric_fn` if `tensors` is list or keyword arguments if `tensors` is
  a dict. `metric_fn` takes the `tensors` and returns a dict from metric string
  name to the result of calling a metric function, namely a `(metric_tensor,
  update_op)` tuple. See `TPUEstimator` for MNIST example how to specify the
  `eval_metrics`.

  `scaffold_fn` is a function running on CPU to generate the `Scaffold`. This
  function should not capture any Tensors in `model_fn`.

  `host_call` is a tuple of a `function` and a list or dictionary of `tensors`
  to pass to that function and returns a list of Tensors. `host_call` currently
  works for train() and evaluate(). The Tensors returned by the function is
  executed on the CPU on every step, so there is communication overhead when
  sending tensors from TPU to CPU. To reduce the overhead, try reducing the
  size of the tensors. The `tensors` are concatenated along their major (batch)
  dimension, and so must be >= rank 1. The `host_call` is useful for writing
  summaries with `tf.contrib.summary.create_file_writer`.
  "
  [ mode predictions loss train_op eval_metrics export_outputs scaffold_fn host_call training_hooks evaluation_hooks prediction_hooks ]
  (py/call-attr tpu "TPUEstimatorSpec"  mode predictions loss train_op eval_metrics export_outputs scaffold_fn host_call training_hooks evaluation_hooks prediction_hooks ))

(defn as-estimator-spec 
  "Creates an equivalent `EstimatorSpec` used by CPU train/eval."
  [ self  ]
  (py/call-attr self "as_estimator_spec"  self  ))

(defn eval-metrics 
  "Alias for field number 4"
  [ self ]
    (py/call-attr self "eval_metrics"))

(defn evaluation-hooks 
  "Alias for field number 9"
  [ self ]
    (py/call-attr self "evaluation_hooks"))

(defn export-outputs 
  "Alias for field number 5"
  [ self ]
    (py/call-attr self "export_outputs"))

(defn host-call 
  "Alias for field number 7"
  [ self ]
    (py/call-attr self "host_call"))

(defn loss 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "loss"))

(defn mode 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "mode"))

(defn prediction-hooks 
  "Alias for field number 10"
  [ self ]
    (py/call-attr self "prediction_hooks"))

(defn predictions 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "predictions"))

(defn scaffold-fn 
  "Alias for field number 6"
  [ self ]
    (py/call-attr self "scaffold_fn"))

(defn train-op 
  "Alias for field number 3"
  [ self ]
    (py/call-attr self "train_op"))

(defn training-hooks 
  "Alias for field number 8"
  [ self ]
    (py/call-attr self "training_hooks"))
