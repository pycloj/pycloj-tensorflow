(ns tensorflow-estimator.python.estimator.api.-v1.estimator.experimental.InMemoryEvaluatorHook
  "Hook to run evaluation in training without a checkpoint.

  Example:

  ```python
  def train_input_fn():
    ...
    return train_dataset

  def eval_input_fn():
    ...
    return eval_dataset

  estimator = tf.estimator.DNNClassifier(...)

  evaluator = tf.estimator.experimental.InMemoryEvaluatorHook(
      estimator, eval_input_fn)
  estimator.train(train_input_fn, hooks=[evaluator])
  ```

  Current limitations of this approach are:

  * It doesn't support multi-node distributed mode.
  * It doesn't support saveable objects other than variables (such as boosted
    tree support)
  * It doesn't support custom saver logic (such as ExponentialMovingAverage
    support)

  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow_estimator.python.estimator.api._v1.estimator.experimental"))
(defn InMemoryEvaluatorHook 
  "Hook to run evaluation in training without a checkpoint.

  Example:

  ```python
  def train_input_fn():
    ...
    return train_dataset

  def eval_input_fn():
    ...
    return eval_dataset

  estimator = tf.estimator.DNNClassifier(...)

  evaluator = tf.estimator.experimental.InMemoryEvaluatorHook(
      estimator, eval_input_fn)
  estimator.train(train_input_fn, hooks=[evaluator])
  ```

  Current limitations of this approach are:

  * It doesn't support multi-node distributed mode.
  * It doesn't support saveable objects other than variables (such as boosted
    tree support)
  * It doesn't support custom saver logic (such as ExponentialMovingAverage
    support)

  "
  [estimator input_fn steps hooks name  & {:keys [every_n_iter]} ]
    (py/call-attr-kw experimental "InMemoryEvaluatorHook" [estimator input_fn steps hooks name] {:every_n_iter every_n_iter }))

(defn after-create-session 
  "Does first run which shows the eval metrics before training."
  [ self session coord ]
  (py/call-attr self "after_create_session"  self session coord ))

(defn after-run 
  "Runs evaluator."
  [ self run_context run_values ]
  (py/call-attr self "after_run"  self run_context run_values ))

(defn before-run 
  "Called before each call to run().

    You can return from this call a `SessionRunArgs` object indicating ops or
    tensors to add to the upcoming `run()` call.  These ops/tensors will be run
    together with the ops/tensors originally passed to the original run() call.
    The run args you return can also contain feeds to be added to the run()
    call.

    The `run_context` argument is a `SessionRunContext` that provides
    information about the upcoming `run()` call: the originally requested
    op/tensors, the TensorFlow Session.

    At this point graph is finalized and you can not add ops.

    Args:
      run_context: A `SessionRunContext` object.

    Returns:
      None or a `SessionRunArgs` object.
    "
  [ self run_context ]
  (py/call-attr self "before_run"  self run_context ))

(defn begin 
  "Build eval graph and restoring op."
  [ self  ]
  (py/call-attr self "begin"  self  ))

(defn end 
  "Runs evaluator for final model."
  [ self session ]
  (py/call-attr self "end"  self session ))
