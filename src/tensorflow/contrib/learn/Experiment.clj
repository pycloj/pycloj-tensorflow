(ns tensorflow.contrib.learn.Experiment
  "Experiment is a class containing all information needed to train a model.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  After an experiment is created (by passing an Estimator and inputs for
  training and evaluation), an Experiment instance knows how to invoke training
  and eval loops in a sensible fashion for distributed training.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce learn (import-module "tensorflow.contrib.learn"))

(defn Experiment 
  "Experiment is a class containing all information needed to train a model.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  After an experiment is created (by passing an Estimator and inputs for
  training and evaluation), an Experiment instance knows how to invoke training
  and eval loops in a sensible fashion for distributed training.
  "
  [estimator train_input_fn eval_input_fn eval_metrics train_steps & {:keys [eval_steps train_monitors eval_hooks local_eval_frequency eval_delay_secs continuous_eval_throttle_secs min_eval_frequency delay_workers_by_global_step export_strategies train_steps_per_iteration checkpoint_and_export saving_listeners check_interval_secs]
                       :or {train_monitors None eval_hooks None local_eval_frequency None min_eval_frequency None export_strategies None train_steps_per_iteration None saving_listeners None}} ]
    (py/call-attr-kw learn "Experiment" [estimator train_input_fn eval_input_fn eval_metrics train_steps] {:eval_steps eval_steps :train_monitors train_monitors :eval_hooks eval_hooks :local_eval_frequency local_eval_frequency :eval_delay_secs eval_delay_secs :continuous_eval_throttle_secs continuous_eval_throttle_secs :min_eval_frequency min_eval_frequency :delay_workers_by_global_step delay_workers_by_global_step :export_strategies export_strategies :train_steps_per_iteration train_steps_per_iteration :checkpoint_and_export checkpoint_and_export :saving_listeners saving_listeners :check_interval_secs check_interval_secs }))

(defn continuous-eval 
  ""
  [self delay_secs throttle_delay_secs & {:keys [evaluate_checkpoint_only_once continuous_eval_predicate_fn name]
                       :or {continuous_eval_predicate_fn None}} ]
    (py/call-attr-kw self "continuous_eval" [delay_secs throttle_delay_secs] {:evaluate_checkpoint_only_once evaluate_checkpoint_only_once :continuous_eval_predicate_fn continuous_eval_predicate_fn :name name }))
(defn continuous-eval-on-train-data 
  ""
  [self delay_secs throttle_delay_secs continuous_eval_predicate_fn  & {:keys [name]} ]
    (py/call-attr-kw self "continuous_eval_on_train_data" [delay_secs throttle_delay_secs continuous_eval_predicate_fn] {:name name }))

(defn continuous-train-and-eval 
  "Interleaves training and evaluation. (experimental)

Warning: THIS FUNCTION IS EXPERIMENTAL. It may change or be removed at any time, and without warning.

The frequency of evaluation is controlled by the `train_steps_per_iteration`
(via constructor). The model will be first trained for
`train_steps_per_iteration`, and then be evaluated in turns.

This method is intended for single machine usage.

This differs from `train_and_evaluate` as follows:

  1. The procedure will have train and evaluation in turns. The model
  will be trained for a number of steps (usually smaller than `train_steps`
  if provided) and then be evaluated.  `train_and_evaluate` will train the
  model for `train_steps` (no small training iterations).

  2. Due to the different approach this schedule takes, it leads to two
  differences in resource control. First, the resources (e.g., memory) used
  by training will be released before evaluation (`train_and_evaluate` takes
  double resources). Second, more checkpoints will be saved as a checkpoint
  is generated at the end of each training iteration.

  3. As the estimator.train starts from scratch (new graph, new states for
  input, etc) at each iteration, it is recommended to have the
  `train_steps_per_iteration` larger. It is also recommended to shuffle your
  input.

Args:
  continuous_eval_predicate_fn: A predicate function determining whether to
    continue eval after each iteration. A `predicate_fn` has one of the
    following signatures:
      * (eval_results) -> boolean
      * (eval_results, checkpoint_path) -> boolean
    Where `eval_results` is the dictionary of metric evaluations and
    checkpoint_path is the path to the checkpoint containing the parameters
    on which that evaluation was based.
    At the beginning of evaluation, the passed `eval_results` and
    `checkpoint_path` will be None so it's expected that the predicate
    function handles that gracefully.
    When `predicate_fn` is not specified, continuous eval will run in an
    infinite loop (if `train_steps` is None). or exit once global step
    reaches `train_steps`.

Returns:
  A tuple of the result of the `evaluate` call to the `Estimator` and the
  export results using the specified `ExportStrategy`.

Raises:
  ValueError: if `continuous_eval_predicate_fn` is neither None nor
    callable."
  [ self continuous_eval_predicate_fn ]
  (py/call-attr self "continuous_train_and_eval"  self continuous_eval_predicate_fn ))

(defn estimator 
  ""
  [ self ]
    (py/call-attr self "estimator"))

(defn eval-metrics 
  ""
  [ self ]
    (py/call-attr self "eval_metrics"))

(defn eval-steps 
  ""
  [ self ]
    (py/call-attr self "eval_steps"))

(defn evaluate 
  "Evaluate on the evaluation data.

    Runs evaluation on the evaluation data and returns the result. Runs for
    `self._eval_steps` steps, or if it's `None`, then run until input is
    exhausted or another exception is raised. Start the evaluation after
    `delay_secs` seconds, or if it's `None`, defaults to using
    `self._eval_delay_secs` seconds.

    Args:
      delay_secs: Start evaluating after this many seconds. If `None`, defaults
        to using `self._eval_delays_secs`.
      name: Gives the name to the evauation for the case multiple evaluation is
        run for the same experiment.

    Returns:
      The result of the `evaluate` call to the `Estimator`.
    "
  [ self delay_secs name ]
  (py/call-attr self "evaluate"  self delay_secs name ))

(defn extend-train-hooks 
  "Extends the hooks for training."
  [ self additional_hooks ]
  (py/call-attr self "extend_train_hooks"  self additional_hooks ))

(defn local-run 
  "DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-23.
Instructions for updating:
local_run will be renamed to train_and_evaluate and the new default behavior will be to run evaluation every time there is a new checkpoint."
  [ self  ]
  (py/call-attr self "local_run"  self  ))

(defn reset-export-strategies 
  "Resets the export strategies with the `new_export_strategies`.

    Args:
      new_export_strategies: A new list of `ExportStrategy`s, or a single one,
        or None.

    Returns:
      The old export strategies.
    "
  [ self new_export_strategies ]
  (py/call-attr self "reset_export_strategies"  self new_export_strategies ))

(defn run-std-server 
  "Starts a TensorFlow server and joins the serving thread.

    Typically used for parameter servers.

    Raises:
      ValueError: if not enough information is available in the estimator's
        config to create a server.
    "
  [ self  ]
  (py/call-attr self "run_std_server"  self  ))

(defn test 
  "Tests training, evaluating and exporting the estimator for a single step.

    Returns:
      The result of the `evaluate` call to the `Estimator`.
    "
  [ self  ]
  (py/call-attr self "test"  self  ))

(defn train 
  "Fit the estimator using the training data.

    Train the estimator for `self._train_steps` steps, after waiting for
    `delay_secs` seconds. If `self._train_steps` is `None`, train forever.

    Args:
      delay_secs: Start training after this many seconds.

    Returns:
      The trained estimator.
    "
  [ self delay_secs ]
  (py/call-attr self "train"  self delay_secs ))

(defn train-and-evaluate 
  "Interleaves training and evaluation.

    The frequency of evaluation is controlled by the constructor arg
    `min_eval_frequency`. When this parameter is 0, evaluation happens
    only after training has completed. Note that evaluation cannot happen
    more frequently than checkpoints are taken. If no new snapshots are
    available when evaluation is supposed to occur, then evaluation doesn't
    happen for another `min_eval_frequency` steps (assuming a checkpoint is
    available at that point). Thus, settings `min_eval_frequency` to 1 means
    that the model will be evaluated everytime there is a new checkpoint.

    This is particular useful for a \"Master\" task in the cloud, whose
    responsibility it is to take checkpoints, evaluate those checkpoints,
    and write out summaries. Participating in training as the supervisor
    allows such a task to accomplish the first and last items, while
    performing evaluation allows for the second.

    Returns:
      The result of the `evaluate` call to the `Estimator` as well as the
      export results using the specified `ExportStrategy`.
    "
  [ self  ]
  (py/call-attr self "train_and_evaluate"  self  ))

(defn train-steps 
  ""
  [ self ]
    (py/call-attr self "train_steps"))
