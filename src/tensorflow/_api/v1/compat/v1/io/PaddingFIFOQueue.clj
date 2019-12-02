(ns tensorflow.-api.v1.compat.v1.io.PaddingFIFOQueue
  "A FIFOQueue that supports batching variable-sized tensors by padding.

  A `PaddingFIFOQueue` may contain components with dynamic shape, while also
  supporting `dequeue_many`.  See the constructor for more details.

  See `tf.queue.QueueBase` for a description of the methods on
  this class.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce io (import-module "tensorflow._api.v1.compat.v1.io"))
(defn PaddingFIFOQueue 
  "A FIFOQueue that supports batching variable-sized tensors by padding.

  A `PaddingFIFOQueue` may contain components with dynamic shape, while also
  supporting `dequeue_many`.  See the constructor for more details.

  See `tf.queue.QueueBase` for a description of the methods on
  this class.
  "
  [capacity dtypes shapes names shared_name  & {:keys [name]} ]
    (py/call-attr-kw io "PaddingFIFOQueue" [capacity dtypes shapes names shared_name] {:name name }))

(defn close 
  "Closes this queue.

    This operation signals that no more elements will be enqueued in
    the given queue. Subsequent `enqueue` and `enqueue_many`
    operations will fail. Subsequent `dequeue` and `dequeue_many`
    operations will continue to succeed if sufficient elements remain
    in the queue. Subsequently dequeue and dequeue_many operations
    that would otherwise block waiting for more elements (if close
    hadn't been called) will now fail immediately.

    If `cancel_pending_enqueues` is `True`, all pending requests will also
    be canceled.

    Args:
      cancel_pending_enqueues: (Optional.) A boolean, defaulting to
        `False` (described above).
      name: A name for the operation (optional).

    Returns:
      The operation that closes the queue.
    "
  [self  & {:keys [cancel_pending_enqueues name]
                       :or {name None}} ]
    (py/call-attr-kw self "close" [] {:cancel_pending_enqueues cancel_pending_enqueues :name name }))

(defn dequeue 
  "Dequeues one element from this queue.

    If the queue is empty when this operation executes, it will block
    until there is an element to dequeue.

    At runtime, this operation may raise an error if the queue is
    `tf.QueueBase.close` before or during its execution. If the
    queue is closed, the queue is empty, and there are no pending
    enqueue operations that can fulfill this request,
    `tf.errors.OutOfRangeError` will be raised. If the session is
    `tf.Session.close`,
    `tf.errors.CancelledError` will be raised.

    Args:
      name: A name for the operation (optional).

    Returns:
      The tuple of tensors that was dequeued.
    "
  [ self name ]
  (py/call-attr self "dequeue"  self name ))

(defn dequeue-many 
  "Dequeues and concatenates `n` elements from this queue.

    This operation concatenates queue-element component tensors along
    the 0th dimension to make a single component tensor.  All of the
    components in the dequeued tuple will have size `n` in the 0th dimension.

    If the queue is closed and there are less than `n` elements left, then an
    `OutOfRange` exception is raised.

    At runtime, this operation may raise an error if the queue is
    `tf.QueueBase.close` before or during its execution. If the
    queue is closed, the queue contains fewer than `n` elements, and
    there are no pending enqueue operations that can fulfill this
    request, `tf.errors.OutOfRangeError` will be raised. If the
    session is `tf.Session.close`,
    `tf.errors.CancelledError` will be raised.

    Args:
      n: A scalar `Tensor` containing the number of elements to dequeue.
      name: A name for the operation (optional).

    Returns:
      The list of concatenated tensors that was dequeued.
    "
  [ self n name ]
  (py/call-attr self "dequeue_many"  self n name ))

(defn dequeue-up-to 
  "Dequeues and concatenates `n` elements from this queue.

    **Note** This operation is not supported by all queues.  If a queue does not
    support DequeueUpTo, then a `tf.errors.UnimplementedError` is raised.

    This operation concatenates queue-element component tensors along
    the 0th dimension to make a single component tensor. If the queue
    has not been closed, all of the components in the dequeued tuple
    will have size `n` in the 0th dimension.

    If the queue is closed and there are more than `0` but fewer than
    `n` elements remaining, then instead of raising a
    `tf.errors.OutOfRangeError` like `tf.QueueBase.dequeue_many`,
    less than `n` elements are returned immediately.  If the queue is
    closed and there are `0` elements left in the queue, then a
    `tf.errors.OutOfRangeError` is raised just like in `dequeue_many`.
    Otherwise the behavior is identical to `dequeue_many`.

    Args:
      n: A scalar `Tensor` containing the number of elements to dequeue.
      name: A name for the operation (optional).

    Returns:
      The tuple of concatenated tensors that was dequeued.
    "
  [ self n name ]
  (py/call-attr self "dequeue_up_to"  self n name ))

(defn dtypes 
  "The list of dtypes for each component of a queue element."
  [ self ]
    (py/call-attr self "dtypes"))

(defn enqueue 
  "Enqueues one element to this queue.

    If the queue is full when this operation executes, it will block
    until the element has been enqueued.

    At runtime, this operation may raise an error if the queue is
    `tf.QueueBase.close` before or during its execution. If the
    queue is closed before this operation runs,
    `tf.errors.CancelledError` will be raised. If this operation is
    blocked, and either (i) the queue is closed by a close operation
    with `cancel_pending_enqueues=True`, or (ii) the session is
    `tf.Session.close`,
    `tf.errors.CancelledError` will be raised.

    Args:
      vals: A tensor, a list or tuple of tensors, or a dictionary containing
        the values to enqueue.
      name: A name for the operation (optional).

    Returns:
      The operation that enqueues a new tuple of tensors to the queue.
    "
  [ self vals name ]
  (py/call-attr self "enqueue"  self vals name ))

(defn enqueue-many 
  "Enqueues zero or more elements to this queue.

    This operation slices each component tensor along the 0th dimension to
    make multiple queue elements. All of the tensors in `vals` must have the
    same size in the 0th dimension.

    If the queue is full when this operation executes, it will block
    until all of the elements have been enqueued.

    At runtime, this operation may raise an error if the queue is
    `tf.QueueBase.close` before or during its execution. If the
    queue is closed before this operation runs,
    `tf.errors.CancelledError` will be raised. If this operation is
    blocked, and either (i) the queue is closed by a close operation
    with `cancel_pending_enqueues=True`, or (ii) the session is
    `tf.Session.close`,
    `tf.errors.CancelledError` will be raised.

    Args:
      vals: A tensor, a list or tuple of tensors, or a dictionary
        from which the queue elements are taken.
      name: A name for the operation (optional).

    Returns:
      The operation that enqueues a batch of tuples of tensors to the queue.
    "
  [ self vals name ]
  (py/call-attr self "enqueue_many"  self vals name ))

(defn from-list 
  "Create a queue using the queue reference from `queues[index]`.

    Args:
      index: An integer scalar tensor that determines the input that gets
        selected.
      queues: A list of `QueueBase` objects.

    Returns:
      A `QueueBase` object.

    Raises:
      TypeError: When `queues` is not a list of `QueueBase` objects,
        or when the data types of `queues` are not all the same.
    "
  [ self index queues ]
  (py/call-attr self "from_list"  self index queues ))

(defn is-closed 
  "Returns true if queue is closed.

    This operation returns true if the queue is closed and false if the queue
    is open.

    Args:
      name: A name for the operation (optional).

    Returns:
      True if the queue is closed and false if the queue is open.
    "
  [ self name ]
  (py/call-attr self "is_closed"  self name ))

(defn name 
  "The name of the underlying queue."
  [ self ]
    (py/call-attr self "name"))

(defn names 
  "The list of names for each component of a queue element."
  [ self ]
    (py/call-attr self "names"))

(defn queue-ref 
  "The underlying queue reference."
  [ self ]
    (py/call-attr self "queue_ref"))

(defn shapes 
  "The list of shapes for each component of a queue element."
  [ self ]
    (py/call-attr self "shapes"))

(defn size 
  "Compute the number of elements in this queue.

    Args:
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this queue.
    "
  [ self name ]
  (py/call-attr self "size"  self name ))
