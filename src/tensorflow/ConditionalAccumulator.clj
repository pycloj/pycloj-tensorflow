(ns tensorflow.ConditionalAccumulator
  "A conditional accumulator for aggregating gradients.

  Up-to-date gradients (i.e., time step at which gradient was computed is
  equal to the accumulator's time step) are added to the accumulator.

  Extraction of the average gradient is blocked until the required number of
  gradients has been accumulated.
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
(defn ConditionalAccumulator 
  "A conditional accumulator for aggregating gradients.

  Up-to-date gradients (i.e., time step at which gradient was computed is
  equal to the accumulator's time step) are added to the accumulator.

  Extraction of the average gradient is blocked until the required number of
  gradients has been accumulated.
  "
  [dtype shape shared_name  & {:keys [name reduction_type]} ]
    (py/call-attr-kw tensorflow "ConditionalAccumulator" [dtype shape shared_name] {:name name :reduction_type reduction_type }))

(defn accumulator-ref 
  "The underlying accumulator reference."
  [ self ]
    (py/call-attr self "accumulator_ref"))

(defn apply-grad 
  "Attempts to apply a gradient to the accumulator.

    The attempt is silently dropped if the gradient is stale, i.e., local_step
    is less than the accumulator's global time step.

    Args:
      grad: The gradient tensor to be applied.
      local_step: Time step at which the gradient was computed.
      name: Optional name for the operation.

    Returns:
      The operation that (conditionally) applies a gradient to the accumulator.

    Raises:
      ValueError: If grad is of the wrong shape
    "
  [self grad & {:keys [local_step name]
                       :or {name None}} ]
    (py/call-attr-kw self "apply_grad" [grad] {:local_step local_step :name name }))

(defn dtype 
  "The datatype of the gradients accumulated by this accumulator."
  [ self ]
    (py/call-attr self "dtype"))

(defn name 
  "The name of the underlying accumulator."
  [ self ]
    (py/call-attr self "name"))

(defn num-accumulated 
  "Number of gradients that have currently been aggregated in accumulator.

    Args:
      name: Optional name for the operation.

    Returns:
      Number of accumulated gradients currently in accumulator.
    "
  [ self name ]
  (py/call-attr self "num_accumulated"  self name ))

(defn set-global-step 
  "Sets the global time step of the accumulator.

    The operation logs a warning if we attempt to set to a time step that is
    lower than the accumulator's own time step.

    Args:
      new_global_step: Value of new time step. Can be a variable or a constant
      name: Optional name for the operation.

    Returns:
      Operation that sets the accumulator's time step.
    "
  [ self new_global_step name ]
  (py/call-attr self "set_global_step"  self new_global_step name ))

(defn take-grad 
  "Attempts to extract the average gradient from the accumulator.

    The operation blocks until sufficient number of gradients have been
    successfully applied to the accumulator.

    Once successful, the following actions are also triggered:

    - Counter of accumulated gradients is reset to 0.
    - Aggregated gradient is reset to 0 tensor.
    - Accumulator's internal time step is incremented by 1.

    Args:
      num_required: Number of gradients that needs to have been aggregated
      name: Optional name for the operation

    Returns:
      A tensor holding the value of the average gradient.

    Raises:
      InvalidArgumentError: If num_required < 1
    "
  [ self num_required name ]
  (py/call-attr self "take_grad"  self num_required name ))
