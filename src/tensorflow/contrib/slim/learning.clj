(ns tensorflow.contrib.slim.python.slim.learning
  "Contains TF-Slim code for training models.

This script contains various functions for training models. These include
manipulating gradients, creating a `train_op` (an operation that computes the
loss and applies the gradients) and a training loop function. The training loop
allows the user to pass in the `train_op` and runs the optimization according
to user-specified arguments. Note that the training loop uses the
tf.compat.v1.train.Supervisor and its managed_session in its implementation to
ensure the
ability of worker processes to recover from failures.

************************************
* A simple working training script *
************************************

  # Load data and create the model:
  images, labels = LoadData(...)
  predictions = MyModel(images)

  # Define the loss:
  slim.losses.log_loss(predictions, labels)
  total_loss = slim.losses.get_total_loss()

  # Define the optimizer:
  optimizer = tf.compat.v1.train.MomentumOptimizer(FLAGS.learning_rate,
  FLAGS.momentum)

  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)

  # Run training.
  slim.learning.train(train_op, my_log_dir)

*************************
* Creating the train_op *
*************************

In order to train, TF-Slim's train loop needs a train_op: an `Operation` that
(a) computes the loss, (b) applies the gradients to update the weights and
(c) returns the value of the loss. slim.learning.create_train_op creates
such an `Operation`. This function also provides the ability to manipulate
the gradients using a few arguments:

  # Create the train_op and clip the gradient norms:
  train_op = slim.learning.create_train_op(
      total_loss,
      optimizer,
      clip_gradient_norm=4)

  # Create the train_op and scale the gradients by providing a map from variable
  # name (or variable) to a scaling coefficient:
  gradient_multipliers = {
    'conv0/weights': 1.2,
    'fc8/weights': 3.4,
  }
  train_op = slim.learning.create_train_op(
      total_loss,
      optimizer,
      gradient_multipliers=gradient_multipliers)

****************************************************************
* Performing additional (non-gradient) updates during training *
****************************************************************

Many networks utilize modules, like BatchNorm, that require performing a series
of non-gradient updates during training. slim.learning.create_train_op allows
a user to pass in a list of update_ops to call along with the gradient updates.

  train_op = slim.learning.create_train_op(total_loss, optimizer, update_ops)

By default, slim.learning.create_train_op includes all update ops that are
part of the `tf.GraphKeys.UPDATE_OPS` collection. Additionally, TF-Slim's
slim.batch_norm function adds the moving mean and moving variance updates to
this collection. Consequently, users who want to use slim.batch_norm will not
need to take any additional steps in order to have the moving mean and moving
variance updates be computed.

However, users with additional, specialized updates can either override the
default update ops or simply add additional update ops to the
`tf.GraphKeys.UPDATE_OPS` collection:

  # Force TF-Slim NOT to use ANY update_ops:
  train_op = slim.learning.create_train_op(
     total_loss,
     optimizer,
     update_ops=[])

  # Use an alternative set of update ops:
  train_op = slim.learning.create_train_op(
     total_loss,
     optimizer,
     update_ops=my_other_update_ops)

  # Use an alternative set of update ops in addition to the default updates:
  tf.compat.v1.add_to_collection(tf.GraphKeys.UPDATE_OPS, my_update0)
  tf.compat.v1.add_to_collection(tf.GraphKeys.UPDATE_OPS, my_update1)

  train_op = slim.learning.create_train_op(
     total_loss,
     optimizer)

  # Which is the same as:
  train_op = slim.learning.create_train_op(
     total_loss,
     optimizer,
     update_ops=tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS))

******************************************
* Initializing a model from a checkpoint *
******************************************

It is common to want to 'warm-start' a model from a pre-trained checkpoint.
TF-Slim provides a convenient mechanism for doing so:

  ...

  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)

  # Create the initial assignment op
  checkpoint_path = '/path/to/old_model_checkpoint'
  variables_to_restore = slim.get_model_variables()
  init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
      checkpoint_path, variables_to_restore)

  # Create an initial assignment function.
  def InitAssignFn(sess):
      sess.run(init_assign_op, init_feed_dict)

  # Run training.
  slim.learning.train(train_op, my_log_dir, init_fn=InitAssignFn)

***************************************************************************
* Initializing a model from a checkpoint whose variable names don't match *
***************************************************************************

At times, a user may want to initialize a new model with values from a
checkpoint whose variable names do not match those of the current model. In this
case, one needs to create a mapping from the checkpoint variable names to the
current model variables. This requires only a small modification of the code
above:
  ...
  # Creates a model with two variables, var0 and var1
  predictions = MyModel(images)
  ...

  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)

  checkpoint_path = '/path/to/old_model_checkpoint'

  # Create the mapping:
  variables_to_restore = {
      'name_var_0_in_checkpoint': slim.get_unique_variable('var0'),
      'name_var_1_in_checkpoint': slim.get_unique_variable('var1')
  }
  init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
      checkpoint_path, variables_to_restore)

  # Create an initial assignment function.
  def InitAssignFn(sess):
      sess.run(init_assign_op, init_feed_dict)

  # Run training.
  slim.learning.train(train_op, my_log_dir, init_fn=InitAssignFn)


*************************************************
* Fine-Tuning Part of a model from a checkpoint *
*************************************************

Rather than initializing all of the weights of a given model, we sometimes
only want to restore some of the weights from a checkpoint. To do this, one
need only filter those variables to initialize as follows:

  ...

  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)

  checkpoint_path = '/path/to/old_model_checkpoint'

  # Specify the variables to restore via a list of inclusion or exclusion
  # patterns:
  variables_to_restore = slim.get_variables_to_restore(
      include=[\"conv\"], exclude=[\"fc8\", \"fc9])
  # or
  variables_to_restore = slim.get_variables_to_restore(exclude=[\"conv\"])

  init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
      checkpoint_path, variables_to_restore)

  # Create an initial assignment function.
  def InitAssignFn(sess):
      sess.run(init_assign_op, init_feed_dict)

  # Run training.
  slim.learning.train(train_op, my_log_dir, init_fn=InitAssignFn)

******************************************************
* Initializing model variables from values in memory *
******************************************************

One may want to initialize the weights of a model from values from an arbitrary
source (a text document, matlab file, etc). While this is technically feasible
using plain TensorFlow, it also results in the values of your weights being
stored in the graph. For large models, this becomes prohibitively large. TF-Slim
allows you to perform this initial assignment without having to store the values
of the initial model in the graph itself by using placeholders and a feed
dictionary:

  ...

  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)

  # Create the mapping from variable names to values:
  var0_initial_value = ReadFromDisk(...)
  var1_initial_value = ReadFromDisk(...)

  var_names_to_values = {
    'var0': var0_initial_value,
    'var1': var1_initial_value,
  }
  init_assign_op, init_feed_dict = slim.assign_from_values(var_names_to_values)

  # Create an initial assignment function.
  def InitAssignFn(sess):
      sess.run(init_assign_op, init_feed_dict)

  # Run training.
  slim.learning.train(train_op, my_log_dir, init_fn=InitAssignFn)
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce learning (import-module "tensorflow.contrib.slim.python.slim.learning"))

(defn add-gradients-summaries 
  "Add summaries to gradients.

  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).

  Returns:
    The list of created summaries.
  "
  [ grads_and_vars ]
  (py/call-attr learning "add_gradients_summaries"  grads_and_vars ))

(defn clip-gradient-norms 
  "Clips the gradients by the given value.

  Args:
    gradients_to_variables: A list of gradient to variable pairs (tuples).
    max_norm: the maximum norm value.

  Returns:
    A list of clipped gradient to variable pairs.
  "
  [ gradients_to_variables max_norm ]
  (py/call-attr learning "clip_gradient_norms"  gradients_to_variables max_norm ))

(defn create-train-op 
  "Creates an `Operation` that evaluates the gradients and returns the loss.

  Args:
    total_loss: A `Tensor` representing the total loss.
    optimizer: A tf.Optimizer to use for computing the gradients.
    global_step: A `Tensor` representing the global step variable. If left as
      `_USE_GLOBAL_STEP`, then tf.contrib.framework.global_step() is used.
    update_ops: An optional list of updates to execute. If `update_ops` is
      `None`, then the update ops are set to the contents of the
      `tf.GraphKeys.UPDATE_OPS` collection. If `update_ops` is not `None`, but
      it doesn't contain all of the update ops in `tf.GraphKeys.UPDATE_OPS`, a
      warning will be displayed.
    variables_to_train: an optional list of variables to train. If None, it will
      default to all tf.compat.v1.trainable_variables().
    clip_gradient_norm: If greater than 0 then the gradients would be clipped by
      it.
    summarize_gradients: Whether or not add summaries for each gradient.
    gate_gradients: How to gate the computation of gradients. See tf.Optimizer.
    aggregation_method: Specifies the method used to combine gradient terms.
      Valid values are defined in the class `AggregationMethod`.
    colocate_gradients_with_ops: Whether or not to try colocating the gradients
      with the ops that generated them.
    gradient_multipliers: A dictionary of either `Variables` or `Variable` op
      names to the coefficient by which the associated gradient should be
      scaled.
    check_numerics: Whether or not we apply check_numerics.

  Returns:
    A `Tensor` that when evaluated, computes the gradients and returns the total
      loss value.
  "
  [total_loss optimizer & {:keys [global_step update_ops variables_to_train clip_gradient_norm summarize_gradients gate_gradients aggregation_method colocate_gradients_with_ops gradient_multipliers check_numerics]
                       :or {update_ops None variables_to_train None aggregation_method None gradient_multipliers None}} ]
    (py/call-attr-kw learning "create_train_op" [total_loss optimizer] {:global_step global_step :update_ops update_ops :variables_to_train variables_to_train :clip_gradient_norm clip_gradient_norm :summarize_gradients summarize_gradients :gate_gradients gate_gradients :aggregation_method aggregation_method :colocate_gradients_with_ops colocate_gradients_with_ops :gradient_multipliers gradient_multipliers :check_numerics check_numerics }))

(defn multiply-gradients 
  "Multiply specified gradients.

  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).
    gradient_multipliers: A map from either `Variables` or `Variable` op names
      to the coefficient by which the associated gradient should be scaled.

  Returns:
    The updated list of gradient to variable pairs.

  Raises:
    ValueError: If `grads_and_vars` is not a list or if `gradient_multipliers`
    is empty or None or if `gradient_multipliers` is not a dictionary.
  "
  [ grads_and_vars gradient_multipliers ]
  (py/call-attr learning "multiply_gradients"  grads_and_vars gradient_multipliers ))

(defn train 
  "Runs a training loop using a TensorFlow supervisor.

  When the sync_optimizer is supplied, gradient updates are applied
  synchronously. Otherwise, gradient updates are applied asynchronous.

  Args:
    train_op: A `Tensor` that, when executed, will apply the gradients and
      return the loss value.
    logdir: The directory where training logs are written to. If None, model
      checkpoints and summaries will not be written.
    train_step_fn: The function to call in order to execute a single gradient
      step. The function must have take exactly four arguments: the current
        session, the `train_op` `Tensor`, a global step `Tensor` and a
        dictionary.
    train_step_kwargs: A dictionary which is passed to the `train_step_fn`. By
      default, two `Boolean`, scalar ops called \"should_stop\" and \"should_log\"
      are provided.
    log_every_n_steps: The frequency, in terms of global steps, that the loss
      and global step are logged.
    graph: The graph to pass to the supervisor. If no graph is supplied the
      default graph is used.
    master: The address of the tensorflow master.
    is_chief: Specifies whether or not the training is being run by the primary
      replica during replica training.
    global_step: The `Tensor` representing the global step. If left as `None`,
      then training_util.get_or_create_global_step(), that is,
      tf.contrib.framework.global_step() is used.
    number_of_steps: The max number of gradient steps to take during training,
      as measured by 'global_step': training will stop if global_step is greater
        than 'number_of_steps'. If the value is left as None, training proceeds
        indefinitely.
    init_op: The initialization operation. If left to its default value, then
      the session is initialized by calling
      `tf.compat.v1.global_variables_initializer()`.
    init_feed_dict: A feed dictionary to use when executing the `init_op`.
    local_init_op: The local initialization operation. If left to its default
      value, then the session is initialized by calling
      `tf.compat.v1.local_variables_initializer()` and
      `tf.compat.v1.tables_initializer()`.
    init_fn: An optional callable to be executed after `init_op` is called. The
      callable must accept one argument, the session being initialized.
    ready_op: Operation to check if the model is ready to use. If left to its
      default value, then the session checks for readiness by calling
      `tf.compat.v1.report_uninitialized_variables()`.
    summary_op: The summary operation.
    save_summaries_secs: How often, in seconds, to save summaries.
    summary_writer: `SummaryWriter` to use.  Can be `None` to indicate that no
      summaries should be written. If unset, we create a SummaryWriter.
    startup_delay_steps: The number of steps to wait for before beginning. Note
      that this must be 0 if a sync_optimizer is supplied.
    saver: Saver to save checkpoints. If None, a default one will be created and
      used.
    save_interval_secs: How often, in seconds, to save the model to `logdir`.
    sync_optimizer: an instance of tf.compat.v1.train.SyncReplicasOptimizer, or
      a list of them. If the argument is supplied, gradient updates will be
      synchronous. If left as `None`, gradient updates will be asynchronous.
    session_config: An instance of `tf.compat.v1.ConfigProto` that will be used
      to configure the `Session`. If left as `None`, the default will be used.
    session_wrapper: A function that takes a `tf.compat.v1.Session` object as
      the only argument and returns a wrapped session object that has the same
      methods that the original object has, or `None`. Iff not `None`, the
      wrapped object will be used for training.
    trace_every_n_steps: produce and save a `Timeline` in Chrome trace format
      and add it to the summaries every `trace_every_n_steps`. If None, no trace
      information will be produced or saved.
    ignore_live_threads: If `True` ignores threads that remain running after a
      grace period when stopping the supervisor, instead of raising a
      RuntimeError.

  Returns:
    the value of the loss function after training.

  Raises:
    ValueError: if `train_op` is empty or if `startup_delay_steps` is
      non-zero when `sync_optimizer` is supplied, if `number_of_steps` is
      negative, or if `trace_every_n_steps` is not `None` and no `logdir` is
      provided.
  "
  [train_op logdir & {:keys [train_step_fn train_step_kwargs log_every_n_steps graph master is_chief global_step number_of_steps init_op init_feed_dict local_init_op init_fn ready_op summary_op save_summaries_secs summary_writer startup_delay_steps saver save_interval_secs sync_optimizer session_config session_wrapper trace_every_n_steps ignore_live_threads]
                       :or {graph None global_step None number_of_steps None init_feed_dict None init_fn None saver None sync_optimizer None session_config None session_wrapper None trace_every_n_steps None}} ]
    (py/call-attr-kw learning "train" [train_op logdir] {:train_step_fn train_step_fn :train_step_kwargs train_step_kwargs :log_every_n_steps log_every_n_steps :graph graph :master master :is_chief is_chief :global_step global_step :number_of_steps number_of_steps :init_op init_op :init_feed_dict init_feed_dict :local_init_op local_init_op :init_fn init_fn :ready_op ready_op :summary_op summary_op :save_summaries_secs save_summaries_secs :summary_writer summary_writer :startup_delay_steps startup_delay_steps :saver saver :save_interval_secs save_interval_secs :sync_optimizer sync_optimizer :session_config session_config :session_wrapper session_wrapper :trace_every_n_steps trace_every_n_steps :ignore_live_threads ignore_live_threads }))

(defn train-step 
  "Function that takes a gradient step and specifies whether to stop.

  Args:
    sess: The current session.
    train_op: An `Operation` that evaluates the gradients and returns the total
      loss.
    global_step: A `Tensor` representing the global training step.
    train_step_kwargs: A dictionary of keyword arguments.

  Returns:
    The total loss and a boolean indicating whether or not to stop training.

  Raises:
    ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
  "
  [ sess train_op global_step train_step_kwargs ]
  (py/call-attr learning "train_step"  sess train_op global_step train_step_kwargs ))
