(ns tensorflow.-api.v1.compat.v1.SparseConditionalAccumulator
  "A conditional accumulator for aggregating sparse gradients.

  Sparse gradients are represented by `IndexedSlices`.

  Up-to-date gradients (i.e., time step at which gradient was computed is
  equal to the accumulator's time step) are added to the accumulator.

  Extraction of the average gradient is blocked until the required number of
  gradients has been accumulated.

  Args:
    dtype: Datatype of the accumulated gradients.
    shape: Shape of the accumulated gradients.
    shared_name: Optional. If non-empty, this accumulator will be shared under
      the given name across multiple sessions.
    name: Optional name for the accumulator.
    reduction_type: Reduction type to use when taking the gradient.
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
(defn SparseConditionalAccumulator 
  "A conditional accumulator for aggregating sparse gradients.

  Sparse gradients are represented by `IndexedSlices`.

  Up-to-date gradients (i.e., time step at which gradient was computed is
  equal to the accumulator's time step) are added to the accumulator.

  Extraction of the average gradient is blocked until the required number of
  gradients has been accumulated.

  Args:
    dtype: Datatype of the accumulated gradients.
    shape: Shape of the accumulated gradients.
    shared_name: Optional. If non-empty, this accumulator will be shared under
      the given name across multiple sessions.
    name: Optional name for the accumulator.
    reduction_type: Reduction type to use when taking the gradient.
  "
  [dtype shape shared_name  & {:keys [name reduction_type]} ]
    (py/call-attr-kw v1 "SparseConditionalAccumulator" [dtype shape shared_name] {:name name :reduction_type reduction_type }))

(defn accumulator-ref 
  "The underlying accumulator reference."
  [ self ]
    (py/call-attr self "accumulator_ref"))

(defn apply-grad 
  "Attempts to apply a sparse gradient to the accumulator.

    The attempt is silently dropped if the gradient is stale, i.e., `local_step`
    is less than the accumulator's global time step.

    A sparse gradient is represented by its indices, values and possibly empty
    or None shape. Indices must be a vector representing the locations of
    non-zero entries in the tensor. Values are the non-zero slices of the
    gradient, and must have the same first dimension as indices, i.e., the nnz
    represented by indices and values must be consistent. Shape, if not empty or
    None, must be consistent with the accumulator's shape (if also provided).

    Example:
      A tensor [[0, 0], [0, 1], [2, 3]] can be represented
        indices: [1,2]
        values: [[0,1],[2,3]]
        shape: [3, 2]

    Args:
      grad_indices: Indices of the sparse gradient to be applied.
      grad_values: Values of the sparse gradient to be applied.
      grad_shape: Shape of the sparse gradient to be applied.
      local_step: Time step at which the gradient was computed.
      name: Optional name for the operation.

    Returns:
      The operation that (conditionally) applies a gradient to the accumulator.

    Raises:
      InvalidArgumentError: If grad is of the wrong shape
    "
  [self grad_indices grad_values grad_shape & {:keys [local_step name]
                       :or {name None}} ]
    (py/call-attr-kw self "apply_grad" [grad_indices grad_values grad_shape] {:local_step local_step :name name }))

(defn apply-indexed-slices-grad 
  "Attempts to apply a gradient to the accumulator.

    The attempt is silently dropped if the gradient is stale, i.e., `local_step`
    is less than the accumulator's global time step.

    Args:
      grad: The gradient `IndexedSlices` to be applied.
      local_step: Time step at which the gradient was computed.
      name: Optional name for the operation.

    Returns:
      The operation that (conditionally) applies a gradient to the accumulator.

    Raises:
      InvalidArgumentError: If grad is of the wrong shape
    "
  [self grad & {:keys [local_step name]
                       :or {name None}} ]
    (py/call-attr-kw self "apply_indexed_slices_grad" [grad] {:local_step local_step :name name }))

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
      A tuple of indices, values, and shape representing the average gradient.

    Raises:
      InvalidArgumentError: If `num_required` < 1
    "
  [ self num_required name ]
  (py/call-attr self "take_grad"  self num_required name ))

(defn take-indexed-slices-grad 
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
      An `IndexedSlices` holding the value of the average gradient.

    Raises:
      InvalidArgumentError: If `num_required` < 1
    "
  [ self num_required name ]
  (py/call-attr self "take_indexed_slices_grad"  self num_required name ))
