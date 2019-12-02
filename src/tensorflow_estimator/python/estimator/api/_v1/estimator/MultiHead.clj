(ns tensorflow-estimator.python.estimator.api.-v1.estimator.MultiHead
  "Creates a `Head` for multi-objective learning.

  This class merges the output of multiple `Head` objects. Specifically:

  * For training, sums losses of each head, calls `train_op_fn` with this
    final loss.
  * For eval, merges metrics by adding `head.name` suffix to the keys in eval
    metrics, such as `precision/head1.name`, `precision/head2.name`.
  * For prediction, merges predictions and updates keys in prediction dict to a
    2-tuple, `(head.name, prediction_key)`. Merges `export_outputs` such that
    by default the first head is served.

  Usage:

  ```python
  # In `input_fn`, specify labels as a dict keyed by head name:
  def input_fn():
    features = ...
    labels1 = ...
    labels2 = ...
    return features, {'head1.name': labels1, 'head2.name': labels2}

  # In `model_fn`, specify logits as a dict keyed by head name:
  def model_fn(features, labels, mode):
    # Create simple heads and specify head name.
    head1 = tf.estimator.MultiClassHead(n_classes=3, name='head1')
    head2 = tf.estimator.BinaryClassHead(name='head2')
    # Create MultiHead from two simple heads.
    head = tf.estimator.MultiHead([head1, head2])
    # Create logits for each head, and combine them into a dict.
    logits1, logits2 = logit_fn()
    logits = {'head1.name': logits1, 'head2.name': logits2}
    # Return the merged EstimatorSpec
    return head.create_estimator_spec(..., logits=logits, ...)

  # Create an estimator with this model_fn.
  estimator = tf.estimator.Estimator(model_fn=model_fn)
  estimator.train(input_fn=input_fn)
  ```

  Also supports `logits` as a `Tensor` of shape
  `[D0, D1, ... DN, logits_dimension]`. It will split the `Tensor` along the
  last dimension and distribute it appropriately among the heads. E.g.:
  # Input logits.
  # logits = np.array([[-1., 1., 2., -2., 2.], [-1.5, 1., -3., 2., -2.]],
                      dtype=np.float32)
  # Suppose head1.logits_dimension = 2 and head2.logits_dimension = 3. After
  # splitting, the result is:
  # logits_dict = {'head1_name': [[-1., 1.], [-1.5, 1.]],
                   'head2_name':  [[2., -2., 2.], [-3., 2., -2.]]}
  Usage:

  ```python
  def model_fn(features, labels, mode):
    # Create simple heads and specify head name.
    head1 = tf.estimator.MultiClassHead(n_classes=3, name='head1')
    head2 = tf.estimator.BinaryClassHead(name='head2')
    # Create multi-head from two simple heads.
    head = tf.estimator.MultiHead([head1, head2])
    # Create logits for the multihead. The result of logits is a `Tensor`.
    logits = logit_fn(logits_dimension=head.logits_dimension)
    # Return the merged EstimatorSpec
    return head.create_estimator_spec(..., logits=logits, ...)
  ```

  Args:
    heads: List or tuple of `Head` instances. All heads must have `name`
      specified. The first head in the list is the default used at serving time.
    head_weights: Optional list of weights, same length as `heads`. Used when
      merging losses to calculate the weighted sum of losses from each head. If
      `None`, all losses are weighted equally.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce estimator (import-module "tensorflow_estimator.python.estimator.api._v1.estimator"))

(defn MultiHead 
  "Creates a `Head` for multi-objective learning.

  This class merges the output of multiple `Head` objects. Specifically:

  * For training, sums losses of each head, calls `train_op_fn` with this
    final loss.
  * For eval, merges metrics by adding `head.name` suffix to the keys in eval
    metrics, such as `precision/head1.name`, `precision/head2.name`.
  * For prediction, merges predictions and updates keys in prediction dict to a
    2-tuple, `(head.name, prediction_key)`. Merges `export_outputs` such that
    by default the first head is served.

  Usage:

  ```python
  # In `input_fn`, specify labels as a dict keyed by head name:
  def input_fn():
    features = ...
    labels1 = ...
    labels2 = ...
    return features, {'head1.name': labels1, 'head2.name': labels2}

  # In `model_fn`, specify logits as a dict keyed by head name:
  def model_fn(features, labels, mode):
    # Create simple heads and specify head name.
    head1 = tf.estimator.MultiClassHead(n_classes=3, name='head1')
    head2 = tf.estimator.BinaryClassHead(name='head2')
    # Create MultiHead from two simple heads.
    head = tf.estimator.MultiHead([head1, head2])
    # Create logits for each head, and combine them into a dict.
    logits1, logits2 = logit_fn()
    logits = {'head1.name': logits1, 'head2.name': logits2}
    # Return the merged EstimatorSpec
    return head.create_estimator_spec(..., logits=logits, ...)

  # Create an estimator with this model_fn.
  estimator = tf.estimator.Estimator(model_fn=model_fn)
  estimator.train(input_fn=input_fn)
  ```

  Also supports `logits` as a `Tensor` of shape
  `[D0, D1, ... DN, logits_dimension]`. It will split the `Tensor` along the
  last dimension and distribute it appropriately among the heads. E.g.:
  # Input logits.
  # logits = np.array([[-1., 1., 2., -2., 2.], [-1.5, 1., -3., 2., -2.]],
                      dtype=np.float32)
  # Suppose head1.logits_dimension = 2 and head2.logits_dimension = 3. After
  # splitting, the result is:
  # logits_dict = {'head1_name': [[-1., 1.], [-1.5, 1.]],
                   'head2_name':  [[2., -2., 2.], [-3., 2., -2.]]}
  Usage:

  ```python
  def model_fn(features, labels, mode):
    # Create simple heads and specify head name.
    head1 = tf.estimator.MultiClassHead(n_classes=3, name='head1')
    head2 = tf.estimator.BinaryClassHead(name='head2')
    # Create multi-head from two simple heads.
    head = tf.estimator.MultiHead([head1, head2])
    # Create logits for the multihead. The result of logits is a `Tensor`.
    logits = logit_fn(logits_dimension=head.logits_dimension)
    # Return the merged EstimatorSpec
    return head.create_estimator_spec(..., logits=logits, ...)
  ```

  Args:
    heads: List or tuple of `Head` instances. All heads must have `name`
      specified. The first head in the list is the default used at serving time.
    head_weights: Optional list of weights, same length as `heads`. Used when
      merging losses to calculate the weighted sum of losses from each head. If
      `None`, all losses are weighted equally.
  "
  [ heads head_weights ]
  (py/call-attr estimator "MultiHead"  heads head_weights ))

(defn create-estimator-spec 
  "Returns a `model_fn.EstimatorSpec`.

    Args:
      features: Input `dict` of `Tensor` or `SparseTensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: Input `dict` keyed by head name, or logits `Tensor` with shape
        `[D0, D1, ... DN, logits_dimension]`. For many applications, the
        `Tensor` shape is `[batch_size, logits_dimension]`. If logits is a
        `Tensor`, it  will split the `Tensor` along the last dimension and
        distribute it appropriately among the heads. Check `MultiHead` for
        examples.
      labels: Input `dict` keyed by head name. For each head, the label value
        can be integer or string `Tensor` with shape matching its corresponding
        `logits`.`labels` is a required argument when `mode` equals `TRAIN` or
        `EVAL`.
      optimizer: An `tf.keras.optimizers.Optimizer` instance to optimize the
        loss in TRAIN mode. Namely, sets `train_op = optimizer.get_updates(loss,
        trainable_variables)`, which updates variables to minimize `loss`.
      trainable_variables: A list or tuple of `Variable` objects to update to
        minimize `loss`. In Tensorflow 1.x, by default these are the list of
        variables collected in the graph under the key
        `GraphKeys.TRAINABLE_VARIABLES`. As Tensorflow 2.x doesn't have
        collections and GraphKeys, trainable_variables need to be passed
        explicitly here.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns
        `train_op`. Used if `optimizer` is `None`.
      update_ops: A list or tuple of update ops to be run at training time. For
        example, layers such as BatchNormalization create mean and variance
        update ops that need to be run at training time. In Tensorflow 1.x,
        these are thrown into an UPDATE_OPS collection. As Tensorflow 2.x
        doesn't have collections, update_ops need to be passed explicitly here.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses. These losses are
        usually expressed as a batch average, so for best results, in each head,
        users need to use the default `loss_reduction=SUM_OVER_BATCH_SIZE` to
        avoid scaling errors.  Compared to the regularization losses for each
        head, this loss is to regularize the merged loss of all heads in multi
        head, and will be added to the overall training loss of multi head.

    Returns:
      A `model_fn.EstimatorSpec` instance.

    Raises:
      ValueError: If both `train_op_fn` and `optimizer` are `None` in TRAIN
      mode, or if both are set.
      If `mode` is not in Estimator's `ModeKeys`.
    "
  [ self features mode logits labels optimizer trainable_variables train_op_fn update_ops regularization_losses ]
  (py/call-attr self "create_estimator_spec"  self features mode logits labels optimizer trainable_variables train_op_fn update_ops regularization_losses ))

(defn logits-dimension 
  "See `base_head.Head` for details."
  [ self ]
    (py/call-attr self "logits_dimension"))

(defn loss 
  "Returns regularized training loss. See `base_head.Head` for details."
  [ self labels logits features mode regularization_losses ]
  (py/call-attr self "loss"  self labels logits features mode regularization_losses ))

(defn loss-reduction 
  "See `base_head.Head` for details."
  [ self ]
    (py/call-attr self "loss_reduction"))

(defn metrics 
  "Creates metrics. See `base_head.Head` for details."
  [ self regularization_losses ]
  (py/call-attr self "metrics"  self regularization_losses ))

(defn name 
  "See `base_head.Head` for details."
  [ self ]
    (py/call-attr self "name"))

(defn predictions 
  "Create predictions. See `base_head.Head` for details."
  [ self logits keys ]
  (py/call-attr self "predictions"  self logits keys ))

(defn update-metrics 
  "Updates eval metrics. See `base_head.Head` for details."
  [ self eval_metrics features logits labels regularization_losses ]
  (py/call-attr self "update_metrics"  self eval_metrics features logits labels regularization_losses ))
