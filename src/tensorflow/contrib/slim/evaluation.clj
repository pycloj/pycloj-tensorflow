(ns tensorflow.contrib.slim.python.slim.evaluation
  "Contains functions for evaluation and summarization of metrics.

The evaluation.py module contains helper functions for evaluating TensorFlow
modules using a variety of metrics and summarizing the results.

**********************
* Evaluating Metrics *
**********************

In the simplest use case, we use a model to create the predictions, then specify
the metrics and choose one model checkpoint, finally call the`evaluation_once`
method:

  # Create model and obtain the predictions:
  images, labels = LoadData(...)
  predictions = MyModel(images)

  # Choose the metrics to compute:
  names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
      \"accuracy\": slim.metrics.accuracy(predictions, labels),
      \"mse\": slim.metrics.mean_squared_error(predictions, labels),
  })

  checkpoint_path = '/tmp/my_model_dir/my_checkpoint'
  log_dir = '/tmp/my_model_eval/'

  initial_op = tf.group(
      tf.compat.v1.global_variables_initializer(),
      tf.compat.v1.local_variables_initializer())

  metric_values = slim.evaluate_once(
      master='',
      checkpoint_path=checkpoint_path,
      log_dir=log_dir,
      num_evals=1,
      initial_op=initial_op,
      eval_op=names_to_updates.values(),
      final_op=name_to_values.values())

  for metric, value in zip(names_to_values.keys(), metric_values):
    logging.info('Metric %s has value: %f', metric, value)

************************************************
* Evaluating a Checkpointed Model with Metrics *
************************************************

Often, one wants to evaluate a model checkpoint saved on disk. This can be
performed once or repeatedly on a set schedule.

To evaluate a particular model, users define zero or more metrics and zero or
more summaries and call the evaluation_loop method:

  # Create model and obtain the predictions:
  images, labels = LoadData(...)
  predictions = MyModel(images)

  # Choose the metrics to compute:
  names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
      \"accuracy\": slim.metrics.accuracy(predictions, labels),
      \"mse\": slim.metrics.mean_squared_error(predictions, labels),
  })

  # Define the summaries to write:
  for metric_name, metric_value in metrics_to_values.iteritems():
    tf.compat.v1.summary.scalar(metric_name, metric_value)

  checkpoint_dir = '/tmp/my_model_dir/'
  log_dir = '/tmp/my_model_eval/'

  # We'll evaluate 1000 batches:
  num_evals = 1000

  # Evaluate every 10 minutes:
  slim.evaluation_loop(
      '',
      checkpoint_dir,
      logdir,
      num_evals=num_evals,
      eval_op=names_to_updates.values(),
      summary_op=tf.contrib.deprecated.merge_summary(summary_ops),
      eval_interval_secs=600)

**************************************************
* Evaluating a Checkpointed Model with Summaries *
**************************************************

At times, an evaluation can be performed without metrics at all but rather
with only summaries. The user need only leave out the 'eval_op' argument:

  # Create model and obtain the predictions:
  images, labels = LoadData(...)
  predictions = MyModel(images)

  # Define the summaries to write:
  tf.compat.v1.summary.scalar(...)
  tf.compat.v1.summary.histogram(...)

  checkpoint_dir = '/tmp/my_model_dir/'
  log_dir = '/tmp/my_model_eval/'

  # Evaluate once every 10 minutes.
  slim.evaluation_loop(
      master='',
      checkpoint_dir,
      logdir,
      num_evals=1,
      summary_op=tf.contrib.deprecated.merge_summary(summary_ops),
      eval_interval_secs=600)

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce evaluation (import-module "tensorflow.contrib.slim.python.slim.evaluation"))

(defn checkpoints-iterator 
  "Continuously yield new checkpoint files as they appear.

  The iterator only checks for new checkpoints when control flow has been
  reverted to it. This means it can miss checkpoints if your code takes longer
  to run between iterations than `min_interval_secs` or the interval at which
  new checkpoints are written.

  The `timeout` argument is the maximum number of seconds to block waiting for
  a new checkpoint.  It is used in combination with the `timeout_fn` as
  follows:

  * If the timeout expires and no `timeout_fn` was specified, the iterator
    stops yielding.
  * If a `timeout_fn` was specified, that function is called and if it returns
    a true boolean value the iterator stops yielding.
  * If the function returns a false boolean value then the iterator resumes the
    wait for new checkpoints.  At this point the timeout logic applies again.

  This behavior gives control to callers on what to do if checkpoints do not
  come fast enough or stop being generated.  For example, if callers have a way
  to detect that the training has stopped and know that no new checkpoints
  will be generated, they can provide a `timeout_fn` that returns `True` when
  the training has stopped.  If they know that the training is still going on
  they return `False` instead.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    min_interval_secs: The minimum number of seconds between yielding
      checkpoints.
    timeout: The maximum number of seconds to wait between checkpoints. If left
      as `None`, then the process will wait indefinitely.
    timeout_fn: Optional function to call after a timeout.  If the function
      returns True, then it means that no new checkpoints will be generated and
      the iterator will exit.  The function is called with no arguments.

  Yields:
    String paths to latest checkpoint files as they arrive.
  "
  [checkpoint_dir & {:keys [min_interval_secs timeout timeout_fn]
                       :or {timeout None timeout_fn None}} ]
    (py/call-attr-kw evaluation "checkpoints_iterator" [checkpoint_dir] {:min_interval_secs min_interval_secs :timeout timeout :timeout_fn timeout_fn }))

(defn evaluate-once 
  "Evaluates the model at the given checkpoint path.

  Args:
    master: The BNS address of the TensorFlow master.
    checkpoint_path: The path to a checkpoint to use for evaluation.
    logdir: The directory where the TensorFlow summaries are written to.
    num_evals: The number of times to run `eval_op`.
    initial_op: An operation run at the beginning of evaluation.
    initial_op_feed_dict: A feed dictionary to use when executing `initial_op`.
    eval_op: A operation run `num_evals` times.
    eval_op_feed_dict: The feed dictionary to use when executing the `eval_op`.
    final_op: An operation to execute after all of the `eval_op` executions. The
      value of `final_op` is returned.
    final_op_feed_dict: A feed dictionary to use when executing `final_op`.
    summary_op: The summary_op to evaluate after running TF-Slims metric ops. By
      default the summary_op is set to tf.compat.v1.summary.merge_all().
    summary_op_feed_dict: An optional feed dictionary to use when running the
      `summary_op`.
    variables_to_restore: A list of TensorFlow variables to restore during
      evaluation. If the argument is left as `None` then
      slim.variables.GetVariablesToRestore() is used.
    session_config: An instance of `tf.compat.v1.ConfigProto` that will be used
      to configure the `Session`. If left as `None`, the default will be used.
    hooks: A list of additional `SessionRunHook` objects to pass during the
      evaluation.

  Returns:
    The value of `final_op` or `None` if `final_op` is `None`.
  "
  [master checkpoint_path logdir & {:keys [num_evals initial_op initial_op_feed_dict eval_op eval_op_feed_dict final_op final_op_feed_dict summary_op summary_op_feed_dict variables_to_restore session_config hooks]
                       :or {initial_op None initial_op_feed_dict None eval_op None eval_op_feed_dict None final_op None final_op_feed_dict None summary_op_feed_dict None variables_to_restore None session_config None hooks None}} ]
    (py/call-attr-kw evaluation "evaluate_once" [master checkpoint_path logdir] {:num_evals num_evals :initial_op initial_op :initial_op_feed_dict initial_op_feed_dict :eval_op eval_op :eval_op_feed_dict eval_op_feed_dict :final_op final_op :final_op_feed_dict final_op_feed_dict :summary_op summary_op :summary_op_feed_dict summary_op_feed_dict :variables_to_restore variables_to_restore :session_config session_config :hooks hooks }))

(defn evaluation-loop 
  "Runs TF-Slim's Evaluation Loop.

  Args:
    master: The BNS address of the TensorFlow master.
    checkpoint_dir: The directory where checkpoints are stored.
    logdir: The directory where the TensorFlow summaries are written to.
    num_evals: The number of times to run `eval_op`.
    initial_op: An operation run at the beginning of evaluation.
    initial_op_feed_dict: A feed dictionary to use when executing `initial_op`.
    init_fn: An optional callable to be executed after `init_op` is called. The
      callable must accept one argument, the session being initialized.
    eval_op: A operation run `num_evals` times.
    eval_op_feed_dict: The feed dictionary to use when executing the `eval_op`.
    final_op: An operation to execute after all of the `eval_op` executions. The
      value of `final_op` is returned.
    final_op_feed_dict: A feed dictionary to use when executing `final_op`.
    summary_op: The summary_op to evaluate after running TF-Slims metric ops. By
      default the summary_op is set to tf.compat.v1.summary.merge_all().
    summary_op_feed_dict: An optional feed dictionary to use when running the
      `summary_op`.
    variables_to_restore: A list of TensorFlow variables to restore during
      evaluation. If the argument is left as `None` then
      slim.variables.GetVariablesToRestore() is used.
    eval_interval_secs: The minimum number of seconds between evaluations.
    max_number_of_evaluations: the max number of iterations of the evaluation.
      If the value is left as 'None', the evaluation continues indefinitely.
    session_config: An instance of `tf.compat.v1.ConfigProto` that will be used
      to configure the `Session`. If left as `None`, the default will be used.
    timeout: The maximum amount of time to wait between checkpoints. If left as
      `None`, then the process will wait indefinitely.
    timeout_fn: Optional function to call after a timeout.  If the function
      returns True, then it means that no new checkpoints will be generated and
      the iterator will exit.  The function is called with no arguments.
    hooks: A list of additional `SessionRunHook` objects to pass during repeated
      evaluations.

  Returns:
    The value of `final_op` or `None` if `final_op` is `None`.
  "
  [master checkpoint_dir logdir & {:keys [num_evals initial_op initial_op_feed_dict init_fn eval_op eval_op_feed_dict final_op final_op_feed_dict summary_op summary_op_feed_dict variables_to_restore eval_interval_secs max_number_of_evaluations session_config timeout timeout_fn hooks]
                       :or {initial_op None initial_op_feed_dict None init_fn None eval_op None eval_op_feed_dict None final_op None final_op_feed_dict None summary_op_feed_dict None variables_to_restore None max_number_of_evaluations None session_config None timeout None timeout_fn None hooks None}} ]
    (py/call-attr-kw evaluation "evaluation_loop" [master checkpoint_dir logdir] {:num_evals num_evals :initial_op initial_op :initial_op_feed_dict initial_op_feed_dict :init_fn init_fn :eval_op eval_op :eval_op_feed_dict eval_op_feed_dict :final_op final_op :final_op_feed_dict final_op_feed_dict :summary_op summary_op :summary_op_feed_dict summary_op_feed_dict :variables_to_restore variables_to_restore :eval_interval_secs eval_interval_secs :max_number_of_evaluations max_number_of_evaluations :session_config session_config :timeout timeout :timeout_fn timeout_fn :hooks hooks }))

(defn wait-for-new-checkpoint 
  "Waits until a new checkpoint file is found.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    last_checkpoint: The last checkpoint path used or `None` if we're expecting
      a checkpoint for the first time.
    seconds_to_sleep: The number of seconds to sleep for before looking for a
      new checkpoint.
    timeout: The maximum number of seconds to wait. If left as `None`, then the
      process will wait indefinitely.

  Returns:
    a new checkpoint path, or None if the timeout was reached.
  "
  [checkpoint_dir last_checkpoint & {:keys [seconds_to_sleep timeout]
                       :or {timeout None}} ]
    (py/call-attr-kw evaluation "wait_for_new_checkpoint" [checkpoint_dir last_checkpoint] {:seconds_to_sleep seconds_to_sleep :timeout timeout }))
