(ns tensorflow.GraphKeys
  "Standard names to use for graph collections.

  The standard library uses various well-known names to collect and
  retrieve values associated with a graph. For example, the
  `tf.Optimizer` subclasses default to optimizing the variables
  collected under `tf.GraphKeys.TRAINABLE_VARIABLES` if none is
  specified, but it is also possible to pass an explicit list of
  variables.

  The following standard keys are defined:

  * `GLOBAL_VARIABLES`: the default collection of `Variable` objects, shared
    across distributed environment (model variables are subset of these). See
    `tf.compat.v1.global_variables`
    for more details.
    Commonly, all `TRAINABLE_VARIABLES` variables will be in `MODEL_VARIABLES`,
    and all `MODEL_VARIABLES` variables will be in `GLOBAL_VARIABLES`.
  * `LOCAL_VARIABLES`: the subset of `Variable` objects that are local to each
    machine. Usually used for temporarily variables, like counters.
    Note: use `tf.contrib.framework.local_variable` to add to this collection.
  * `MODEL_VARIABLES`: the subset of `Variable` objects that are used in the
    model for inference (feed forward). Note: use
    `tf.contrib.framework.model_variable` to add to this collection.
  * `TRAINABLE_VARIABLES`: the subset of `Variable` objects that will
    be trained by an optimizer. See
    `tf.compat.v1.trainable_variables`
    for more details.
  * `SUMMARIES`: the summary `Tensor` objects that have been created in the
    graph. See
    `tf.compat.v1.summary.merge_all`
    for more details.
  * `QUEUE_RUNNERS`: the `QueueRunner` objects that are used to
    produce input for a computation. See
    `tf.compat.v1.train.start_queue_runners`
    for more details.
  * `MOVING_AVERAGE_VARIABLES`: the subset of `Variable` objects that will also
    keep moving averages.  See
    `tf.compat.v1.moving_average_variables`
    for more details.
  * `REGULARIZATION_LOSSES`: regularization losses collected during graph
    construction.

  The following standard keys are _defined_, but their collections are **not**
  automatically populated as many of the others are:

  * `WEIGHTS`
  * `BIASES`
  * `ACTIVATIONS`
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tensorflow (import-module "tensorflow"))

(defn GraphKeys 
  "Standard names to use for graph collections.

  The standard library uses various well-known names to collect and
  retrieve values associated with a graph. For example, the
  `tf.Optimizer` subclasses default to optimizing the variables
  collected under `tf.GraphKeys.TRAINABLE_VARIABLES` if none is
  specified, but it is also possible to pass an explicit list of
  variables.

  The following standard keys are defined:

  * `GLOBAL_VARIABLES`: the default collection of `Variable` objects, shared
    across distributed environment (model variables are subset of these). See
    `tf.compat.v1.global_variables`
    for more details.
    Commonly, all `TRAINABLE_VARIABLES` variables will be in `MODEL_VARIABLES`,
    and all `MODEL_VARIABLES` variables will be in `GLOBAL_VARIABLES`.
  * `LOCAL_VARIABLES`: the subset of `Variable` objects that are local to each
    machine. Usually used for temporarily variables, like counters.
    Note: use `tf.contrib.framework.local_variable` to add to this collection.
  * `MODEL_VARIABLES`: the subset of `Variable` objects that are used in the
    model for inference (feed forward). Note: use
    `tf.contrib.framework.model_variable` to add to this collection.
  * `TRAINABLE_VARIABLES`: the subset of `Variable` objects that will
    be trained by an optimizer. See
    `tf.compat.v1.trainable_variables`
    for more details.
  * `SUMMARIES`: the summary `Tensor` objects that have been created in the
    graph. See
    `tf.compat.v1.summary.merge_all`
    for more details.
  * `QUEUE_RUNNERS`: the `QueueRunner` objects that are used to
    produce input for a computation. See
    `tf.compat.v1.train.start_queue_runners`
    for more details.
  * `MOVING_AVERAGE_VARIABLES`: the subset of `Variable` objects that will also
    keep moving averages.  See
    `tf.compat.v1.moving_average_variables`
    for more details.
  * `REGULARIZATION_LOSSES`: regularization losses collected during graph
    construction.

  The following standard keys are _defined_, but their collections are **not**
  automatically populated as many of the others are:

  * `WEIGHTS`
  * `BIASES`
  * `ACTIVATIONS`
  "
  [  ]
  (py/call-attr tensorflow "GraphKeys"  ))
