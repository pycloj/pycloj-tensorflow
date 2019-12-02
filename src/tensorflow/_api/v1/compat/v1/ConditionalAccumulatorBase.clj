(ns tensorflow.-api.v1.compat.v1.ConditionalAccumulatorBase
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
(defonce v1 (import-module "tensorflow._api.v1.compat.v1"))

(defn ConditionalAccumulatorBase 
  "A conditional accumulator for aggregating gradients.

  Up-to-date gradients (i.e., time step at which gradient was computed is
  equal to the accumulator's time step) are added to the accumulator.

  Extraction of the average gradient is blocked until the required number of
  gradients has been accumulated.
  "
  [ dtype shape accumulator_ref ]
  (py/call-attr v1 "ConditionalAccumulatorBase"  dtype shape accumulator_ref ))

(defn accumulator-ref 
  "The underlying accumulator reference."
  [ self ]
    (py/call-attr self "accumulator_ref"))

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
