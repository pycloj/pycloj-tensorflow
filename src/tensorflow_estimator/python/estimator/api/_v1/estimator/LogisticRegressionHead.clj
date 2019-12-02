(ns tensorflow-estimator.python.estimator.api.-v1.estimator.LogisticRegressionHead
  "Creates a `Head` for logistic regression.

  Uses `sigmoid_cross_entropy_with_logits` loss, which is the same as
  `BinaryClassHead`. The differences compared to `BinaryClassHead` are:

  * Does not support `label_vocabulary`. Instead, labels must be float in the
    range [0, 1].
  * Does not calculate some metrics that do not make sense, such as AUC.
  * In `PREDICT` mode, only returns logits and predictions
    (`=tf.sigmoid(logits)`), whereas `BinaryClassHead` also returns
    probabilities, classes, and class_ids.
  * Export output defaults to `RegressionOutput`, whereas `BinaryClassHead`
    defaults to `PredictOutput`.

  The head expects `logits` with shape `[D0, D1, ... DN, 1]`.
  In many applications, the shape is `[batch_size, 1]`.

  The `labels` shape must match `logits`, namely
  `[D0, D1, ... DN]` or `[D0, D1, ... DN, 1]`.

  If `weight_column` is specified, weights must be of shape
  `[D0, D1, ... DN]` or `[D0, D1, ... DN, 1]`.

  This is implemented as a generalized linear model, see
  https://en.wikipedia.org/wiki/Generalized_linear_model.

  The head can be used with a canned estimator. Example:

  ```python
  my_head = tf.estimator.LogisticRegressionHead()
  my_estimator = tf.estimator.DNNEstimator(
      head=my_head,
      hidden_units=...,
      feature_columns=...)
  ```

  It can also be used with a custom `model_fn`. Example:

  ```python
  def _my_model_fn(features, labels, mode):
    my_head = tf.estimator.LogisticRegressionHead()
    logits = tf.keras.Model(...)(features)

    return my_head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        optimizer=tf.keras.optimizers.Adagrad(lr=0.1),
        logits=logits)

  my_estimator = tf.estimator.Estimator(model_fn=_my_model_fn)
  ```

  Args:
    weight_column: A string or a `NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    loss_reduction: One of `tf.losses.Reduction` except `NONE`. Decides how to
      reduce training loss over batch and label dimension. Defaults to
      `SUM_OVER_BATCH_SIZE`, namely weighted sum of losses divided by
      `batch size * label_dimension`.
    name: name of the head. If provided, summary and metrics keys will be
      suffixed by `\"/\" + name`. Also used as `name_scope` when creating ops.
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

(defn LogisticRegressionHead 
  "Creates a `Head` for logistic regression.

  Uses `sigmoid_cross_entropy_with_logits` loss, which is the same as
  `BinaryClassHead`. The differences compared to `BinaryClassHead` are:

  * Does not support `label_vocabulary`. Instead, labels must be float in the
    range [0, 1].
  * Does not calculate some metrics that do not make sense, such as AUC.
  * In `PREDICT` mode, only returns logits and predictions
    (`=tf.sigmoid(logits)`), whereas `BinaryClassHead` also returns
    probabilities, classes, and class_ids.
  * Export output defaults to `RegressionOutput`, whereas `BinaryClassHead`
    defaults to `PredictOutput`.

  The head expects `logits` with shape `[D0, D1, ... DN, 1]`.
  In many applications, the shape is `[batch_size, 1]`.

  The `labels` shape must match `logits`, namely
  `[D0, D1, ... DN]` or `[D0, D1, ... DN, 1]`.

  If `weight_column` is specified, weights must be of shape
  `[D0, D1, ... DN]` or `[D0, D1, ... DN, 1]`.

  This is implemented as a generalized linear model, see
  https://en.wikipedia.org/wiki/Generalized_linear_model.

  The head can be used with a canned estimator. Example:

  ```python
  my_head = tf.estimator.LogisticRegressionHead()
  my_estimator = tf.estimator.DNNEstimator(
      head=my_head,
      hidden_units=...,
      feature_columns=...)
  ```

  It can also be used with a custom `model_fn`. Example:

  ```python
  def _my_model_fn(features, labels, mode):
    my_head = tf.estimator.LogisticRegressionHead()
    logits = tf.keras.Model(...)(features)

    return my_head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        optimizer=tf.keras.optimizers.Adagrad(lr=0.1),
        logits=logits)

  my_estimator = tf.estimator.Estimator(model_fn=_my_model_fn)
  ```

  Args:
    weight_column: A string or a `NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    loss_reduction: One of `tf.losses.Reduction` except `NONE`. Decides how to
      reduce training loss over batch and label dimension. Defaults to
      `SUM_OVER_BATCH_SIZE`, namely weighted sum of losses divided by
      `batch size * label_dimension`.
    name: name of the head. If provided, summary and metrics keys will be
      suffixed by `\"/\" + name`. Also used as `name_scope` when creating ops.
  "
  [weight_column & {:keys [loss_reduction name]
                       :or {name None}} ]
    (py/call-attr-kw estimator "LogisticRegressionHead" [weight_column] {:loss_reduction loss_reduction :name name }))

(defn create-estimator-spec 
  "Returns `EstimatorSpec` that a model_fn can return.

    It is recommended to pass all args via name.

    Args:
      features: Input `dict` mapping string feature names to `Tensor` or
        `SparseTensor` objects containing the values for that feature in a
        minibatch. Often to be used to fetch example-weight tensor.
      mode: Estimator's `ModeKeys`.
      logits: Logits `Tensor` to be used by the head.
      labels: Labels `Tensor`, or `dict` mapping string label names to `Tensor`
        objects of the label values.
      optimizer: An `tf.keras.optimizers.Optimizer` instance to optimize the
        loss in TRAIN mode. Namely, sets `train_op = optimizer.get_updates(loss,
        trainable_variables)`, which updates variables to minimize `loss`.
      trainable_variables: A list or tuple of `Variable` objects to update to
        minimize `loss`. In Tensorflow 1.x, by default these are the list of
        variables collected in the graph under the key
        `GraphKeys.TRAINABLE_VARIABLES`. As Tensorflow 2.x doesn't have
        collections and GraphKeys, trainable_variables need to be passed
        explicitly here.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns an op
        to optimize the model with the loss in TRAIN mode. Used if `optimizer`
        is `None`. Exactly one of `train_op_fn` and `optimizer` must be set in
        TRAIN mode. By default, it is `None` in other modes. If you want to
        optimize loss yourself, you can pass `lambda _: tf.no_op()` and then use
        `EstimatorSpec.loss` to compute and apply gradients.
      update_ops: A list or tuple of update ops to be run at training time. For
        example, layers such as BatchNormalization create mean and variance
        update ops that need to be run at training time. In Tensorflow 1.x,
        these are thrown into an UPDATE_OPS collection. As Tensorflow 2.x
        doesn't have collections, update_ops need to be passed explicitly here.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses.

    Returns:
      `EstimatorSpec`.
    "
  [ self features mode logits labels optimizer trainable_variables train_op_fn update_ops regularization_losses ]
  (py/call-attr self "create_estimator_spec"  self features mode logits labels optimizer trainable_variables train_op_fn update_ops regularization_losses ))

(defn logits-dimension 
  "See `base_head.Head` for details."
  [ self ]
    (py/call-attr self "logits_dimension"))

(defn loss 
  "Return predictions based on keys. See `base_head.Head` for details."
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
  "Return predictions based on keys.  See `base_head.Head` for details.

    Args:
      logits: logits `Tensor` with shape `[D0, D1, ... DN, logits_dimension]`.
        For many applications, the shape is `[batch_size, logits_dimension]`.

    Returns:
      A dict of predictions.
    "
  [ self logits ]
  (py/call-attr self "predictions"  self logits ))

(defn update-metrics 
  "Updates eval metrics. See `base_head.Head` for details."
  [ self eval_metrics features logits labels regularization_losses ]
  (py/call-attr self "update_metrics"  self eval_metrics features logits labels regularization_losses ))
