(ns tensorflow.contrib.slim.python.slim.data.prefetch-queue
  "Implements a simple prefetch_queue."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce prefetch-queue (import-module "tensorflow.contrib.slim.python.slim.data.prefetch_queue"))

(defn prefetch-queue 
  "Creates a queue to prefetch tensors from `tensors`.

  A queue runner for enqueuing tensors into the prefetch_queue is automatically
  added to the TF QueueRunners collection.

  Example:
  This is for example useful to pre-assemble input batches read with
  `tf.compat.v1.train.batch()` and enqueue the pre-assembled batches.  Ops that
  dequeue
  from the pre-assembled queue will not pay the cost of assembling the batch.

  images, labels = tf.compat.v1.train.batch([image, label], batch_size=32,
  num_threads=4)
  batch_queue = prefetch_queue([images, labels])
  images, labels = batch_queue.dequeue()
  logits = Net(images)
  loss = Loss(logits, labels)

  Args:
    tensors: A list or dictionary of `Tensors` to enqueue in the buffer.
    capacity: An integer. The maximum number of elements in the queue.
    num_threads: An integer.  Number of threads running the enqueue op.
    dynamic_pad: Boolean.  Whether to allow variable dimensions in input shapes.
    shared_name: (optional). If set, this queue will be shared under the given
      name across multiple sessions.
    name: (Optional) A name for the operations.

  Returns:
    A queue from which you can dequeue tensors with the same type and shape
    as `tensors`.
  "
  [tensors & {:keys [capacity num_threads dynamic_pad shared_name name]
                       :or {shared_name None name None}} ]
    (py/call-attr-kw prefetch-queue "prefetch_queue" [tensors] {:capacity capacity :num_threads num_threads :dynamic_pad dynamic_pad :shared_name shared_name :name name }))
