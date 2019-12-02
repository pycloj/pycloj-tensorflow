(ns tensorflow.contrib.distribute.StandardSingleLossStep
  "A step function that implements a training step for a feed forward network.

  An instance of this class is intended to be used as a callable:

  ```python
  ...
  step = step_fn.StandardSingleLossStep(
      dataset, loss_fn, optimizer, distribution)

  # Run a single training step on a given DistributionStrategy:
  step(distribution)
  ...
  ```

  Args:
    dataset_fn: a function that returns a tf.data Dataset that produces the
      input for the model.
    loss_fn: a function that takes a context and inputs as arguments. It returns
      the loss for those inputs. `context` is an instance of
      `values.MultiStepContext` that will be passed when `loss_fn` is run.
      `context` can be used to specify the outputs to be returned from
      `loss_fn`, among other things.
    optimizer: an optimizer that implements an update rule.
    distribution: a `DistributionStrategy` object.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distribute (import-module "tensorflow.contrib.distribute"))
(defn StandardSingleLossStep 
  "A step function that implements a training step for a feed forward network.

  An instance of this class is intended to be used as a callable:

  ```python
  ...
  step = step_fn.StandardSingleLossStep(
      dataset, loss_fn, optimizer, distribution)

  # Run a single training step on a given DistributionStrategy:
  step(distribution)
  ...
  ```

  Args:
    dataset_fn: a function that returns a tf.data Dataset that produces the
      input for the model.
    loss_fn: a function that takes a context and inputs as arguments. It returns
      the loss for those inputs. `context` is an instance of
      `values.MultiStepContext` that will be passed when `loss_fn` is run.
      `context` can be used to specify the outputs to be returned from
      `loss_fn`, among other things.
    optimizer: an optimizer that implements an update rule.
    distribution: a `DistributionStrategy` object.
  "
  [dataset_fn loss_fn optimizer distribution  & {:keys [iterations_per_step]} ]
    (py/call-attr-kw distribute "StandardSingleLossStep" [dataset_fn loss_fn optimizer distribution] {:iterations_per_step iterations_per_step }))

(defn distribution 
  ""
  [ self ]
    (py/call-attr self "distribution"))

(defn initialize 
  ""
  [ self  ]
  (py/call-attr self "initialize"  self  ))
