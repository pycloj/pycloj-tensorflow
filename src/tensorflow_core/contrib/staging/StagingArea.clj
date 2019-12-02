(ns tensorflow-core.contrib.staging.StagingArea
  "Class for staging inputs. No ordering guarantees.

  A `StagingArea` is a TensorFlow data structure that stores tensors across
  multiple steps, and exposes operations that can put and get tensors.

  Each `StagingArea` element is a tuple of one or more tensors, where each
  tuple component has a static dtype, and may have a static shape.

  The capacity of a `StagingArea` may be bounded or unbounded.
  It supports multiple concurrent producers and consumers; and
  provides exactly-once delivery.

  Each element of a `StagingArea` is a fixed-length tuple of tensors whose
  dtypes are described by `dtypes`, and whose shapes are optionally described
  by the `shapes` argument.

  If the `shapes` argument is specified, each component of a staging area
  element must have the respective fixed shape. If it is
  unspecified, different elements may have different shapes,

  It can be configured with a capacity in which case
  put(values) will block until space becomes available.

  Similarly, it can be configured with a memory limit which
  will block put(values) until space is available.
  This is mostly useful for limiting the number of tensors on
  devices such as GPUs.

  All get() and peek() commands block if the requested data
  is not present in the Staging Area.

  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce staging (import-module "tensorflow_core.contrib.staging"))
(defn StagingArea 
  "Class for staging inputs. No ordering guarantees.

  A `StagingArea` is a TensorFlow data structure that stores tensors across
  multiple steps, and exposes operations that can put and get tensors.

  Each `StagingArea` element is a tuple of one or more tensors, where each
  tuple component has a static dtype, and may have a static shape.

  The capacity of a `StagingArea` may be bounded or unbounded.
  It supports multiple concurrent producers and consumers; and
  provides exactly-once delivery.

  Each element of a `StagingArea` is a fixed-length tuple of tensors whose
  dtypes are described by `dtypes`, and whose shapes are optionally described
  by the `shapes` argument.

  If the `shapes` argument is specified, each component of a staging area
  element must have the respective fixed shape. If it is
  unspecified, different elements may have different shapes,

  It can be configured with a capacity in which case
  put(values) will block until space becomes available.

  Similarly, it can be configured with a memory limit which
  will block put(values) until space is available.
  This is mostly useful for limiting the number of tensors on
  devices such as GPUs.

  All get() and peek() commands block if the requested data
  is not present in the Staging Area.

  "
  [dtypes shapes names shared_name  & {:keys [capacity memory_limit]} ]
    (py/call-attr-kw staging "StagingArea" [dtypes shapes names shared_name] {:capacity capacity :memory_limit memory_limit }))

(defn capacity 
  "The maximum number of elements of this staging area."
  [ self ]
    (py/call-attr self "capacity"))

(defn clear 
  "Clears the staging area.

    Args:
        name: A name for the operation (optional)

    Returns:
        The created op
    "
  [ self name ]
  (py/call-attr self "clear"  self name ))

(defn dtypes 
  "The list of dtypes for each component of a staging area element."
  [ self ]
    (py/call-attr self "dtypes"))

(defn get 
  "Gets one element from this staging area.

    If the staging area is empty when this operation executes, it will block
    until there is an element to dequeue.

    Note that unlike others ops that can block, like the queue Dequeue
    operations, this can stop other work from happening.  To avoid this, the
    intended use is for this to be called only when there will be an element
    already available.  One method for doing this in a training loop would be to
    run a `put()` call during a warmup session.run call, and then call both
    `get()` and `put()` in each subsequent step.

    The placement of the returned tensor will be determined by the current
    device scope when this function is called.

    Args:
      name: A name for the operation (optional).

    Returns:
      The tuple of tensors that was gotten.
    "
  [ self name ]
  (py/call-attr self "get"  self name ))

(defn memory-limit 
  "The maximum number of bytes of this staging area."
  [ self ]
    (py/call-attr self "memory_limit"))

(defn name 
  "The name of the staging area."
  [ self ]
    (py/call-attr self "name"))

(defn names 
  "The list of names for each component of a staging area element."
  [ self ]
    (py/call-attr self "names"))

(defn peek 
  "Peeks at an element in the staging area.

    If the staging area is too small to contain the element at
    the specified index, it will block until enough elements
    are inserted to complete the operation.

    The placement of the returned tensor will be determined by
    the current device scope when this function is called.

    Args:
      index: The index of the tensor within the staging area
              to look up.
      name: A name for the operation (optional).

    Returns:
      The tuple of tensors that was gotten.
    "
  [ self index name ]
  (py/call-attr self "peek"  self index name ))

(defn put 
  "Create an op that places a value into the staging area.

    This operation will block if the `StagingArea` has reached
    its capacity.

    Args:
      values: A single tensor, a list or tuple of tensors, or a dictionary with
        tensor values. The number of elements must match the length of the
        list provided to the dtypes argument when creating the StagingArea.
      name: A name for the operation (optional).

    Returns:
        The created op.

    Raises:
      ValueError: If the number or type of inputs don't match the staging area.
    "
  [ self values name ]
  (py/call-attr self "put"  self values name ))

(defn shapes 
  "The list of shapes for each component of a staging area element."
  [ self ]
    (py/call-attr self "shapes"))

(defn size 
  "Returns the number of elements in the staging area.

    Args:
        name: A name for the operation (optional)

    Returns:
        The created op
    "
  [ self name ]
  (py/call-attr self "size"  self name ))
