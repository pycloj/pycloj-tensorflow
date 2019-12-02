(ns tensorflow.contrib.tpu.InfeedQueue
  "A helper object to build a device infeed queue.

  The InfeedQueue builds the host-side and device-side Ops to enqueue and
  dequeue elements, respectively, and ensures that their types and
  shapes match.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tpu (import-module "tensorflow.contrib.tpu"))

(defn InfeedQueue 
  "A helper object to build a device infeed queue.

  The InfeedQueue builds the host-side and device-side Ops to enqueue and
  dequeue elements, respectively, and ensures that their types and
  shapes match.
  "
  [ number_of_tuple_elements tuple_types tuple_shapes shard_dimensions name ]
  (py/call-attr tpu "InfeedQueue"  number_of_tuple_elements tuple_types tuple_shapes shard_dimensions name ))

(defn freeze 
  "Freezes the InfeedQueue so it can no longer be modified.

    The configuration is implicitly frozen before any host-side or
    device-side Ops are generated. The configuration cannot be frozen
    until the types and shapes of the tuple elements have been set.

    Raises:
      ValueError: if the types or shapes of the tuple elements have not been
      set.
    "
  [ self  ]
  (py/call-attr self "freeze"  self  ))
(defn generate-dequeue-op 
  "Generates the device-side Op to dequeue a tuple from the queue.

    Implicitly freezes the queue configuration if it is not already
    frozen, which will raise errors if the shapes and types have not
    been fully specified.

    Args:
      tpu_device: The TPU device ordinal where the infeed instruction should be
        placed. If None, no explicit placement will be performed, and it is up
        to the user to call this API from within a proper TPU device scope.
        The XLA code will fail if the TPU dequeue instruction is not bound to
        any device.

    Returns:
      A list of Outputs corresponding to a shard of infeed dequeued
      into XLA, suitable for use within a replicated block.

    Raises:
      ValueError: if the types or shapes of the tuple elements have not been
      set; or if a dequeue op has already been generated.
    "
  [self   & {:keys [tpu_device]} ]
    (py/call-attr-kw self "generate_dequeue_op" [] {:tpu_device tpu_device }))

(defn generate-enqueue-ops 
  "Generates the host-side Ops to enqueue the shards of a tuple.

    sharded_inputs is a list, one for each shard, of lists of
    Tensors. sharded_inputs[0] is the tuple of Tensors to use to feed
    shard 0 if the queue. Returns the host-side Ops that must be run to
    enqueue the sharded tuple. The Op for shard i is colocated with the inputs
    for shard i.

    Implicitly freezes the queue configuration if it is not already
    frozen. If the configuration has already been frozen, and is not
    compatible with the types and shapes of sharded_inputs, an error
    will be raised.

    Args:
      sharded_inputs: a list of lists of Tensors. The length of the outer list
        determines the number of shards. Each inner list indicates the types
        and shapes of the tuples in the corresponding shard.
      tpu_ordinal_function: if not None, a function that takes the
        shard index as input and returns the ordinal of the TPU device
        the shard's infeed should be placed on. tpu_ordinal_function must be
        set if the inputs are placed on CPU devices.
      placement_function: if not None, a function that takes the shard index as
        input and returns the host device where the enqueue op should be placed
        on.

    Returns:
      A list of host-side Ops, one for each shard, that when executed together
      will enqueue a full-size element of infeed.

    Raises:
      ValueError: if the queue configuration has previously been frozen and the
        shapes of the elements of sharded_inputs are not compatible with the
        frozen configuration; or if the shapes of the elements of sharded_inputs
        don't form a consistent unsharded tuple; or if the elements of a tuple
        have different device constraints.
      TypeError: if the queue configuration has previously been frozen and the
        types of the elements of sharded_inputs are not compatible with the
        frozen configuration; or if the types of the elements of sharded_inputs
        don't form a consistent unsharded tuple.
    "
  [ self sharded_inputs tpu_ordinal_function placement_function ]
  (py/call-attr self "generate_enqueue_ops"  self sharded_inputs tpu_ordinal_function placement_function ))

(defn number-of-shards 
  "Gets the number of shards to use for the InfeedQueue.

    Returns:
      Number of shards or None if the number of shards has not been set.
    "
  [ self ]
    (py/call-attr self "number_of_shards"))

(defn number-of-tuple-elements 
  "Returns the number of InfeedQueue tuple elements."
  [ self ]
    (py/call-attr self "number_of_tuple_elements"))

(defn set-configuration-from-input-tensors 
  "Sets the shapes and types of the queue tuple elements.

    input_tensors is a list of Tensors whose types and shapes are used
    to set the queue configuration.

    Args:
      input_tensors: list of Tensors of the same types and shapes as
        the desired queue Tuple.

    Raises:
      ValueError: if input_tensors is not a list of length
        self.number_of_tuple_elements
    "
  [ self input_tensors ]
  (py/call-attr self "set_configuration_from_input_tensors"  self input_tensors ))

(defn set-configuration-from-sharded-input-tensors 
  "Sets the shapes and types of the queue tuple elements.

    input_tensors is a list of lists of Tensors whose types and shapes are used
    to set the queue configuration. The length of the outer list is the number
    of shards required, and each inner list is the tuple of Tensors to use to
    determine the types and shapes of the corresponding shard. This method
    depends on the shard dimension, and calling it freezes the shard policy.

    Args:
      input_tensors: list of lists of Tensors. The outer list length corresponds
        to the desired number of shards, and each inner list is the size
        and shape of the desired configuration of the corresponding shard.

    Raises:
      ValueError: if any inner list is not a list of length
        self.number_of_tuple_elements; or the inner lists do not combine to
        form a consistent unsharded shape.
      TypeError: if the types of the Tensors in the inner lists do not match.
    "
  [ self input_tensors ]
  (py/call-attr self "set_configuration_from_sharded_input_tensors"  self input_tensors ))

(defn set-number-of-shards 
  "Sets the number of shards to use for the InfeedQueue.

    Args:
      number_of_shards: number of ways to shard the InfeedQueue.

    Raises:
      ValueError: if number_of_shards is not > 0; or the policies have
        been frozen and number_of_shards was already set to something
        else.
    "
  [ self number_of_shards ]
  (py/call-attr self "set_number_of_shards"  self number_of_shards ))

(defn set-shard-dimensions 
  "Sets the shard_dimension of each element of the queue.

    shard_dimensions must be a list of length
    self.number_of_tuple_elements, and each element must be
    convertible to a Dimension compatible with self.tuple_shapes.

    Args:
      shard_dimensions: the dimensions of each queue element.

    Raises:
      ValueError: if shard_dimensions is not of length
        self.number_of_tuple_elements; or an element of
        shard_dimensions cannot be converted to a Dimension; or an
        element of shard_dimensions is a Dimension that is out of
        range for the corresponding tuple element shape.
    "
  [ self shard_dimensions ]
  (py/call-attr self "set_shard_dimensions"  self shard_dimensions ))

(defn set-tuple-shapes 
  "Sets the shape of each element of the queue.

    tuple_shapes must be a list of length
    self.number_of_tuple_elements, and each element must be
    convertible to a TensorShape.

    Args:
      tuple_shapes: the shapes of each queue element.

    Raises:
      ValueError: if tuple_shapes is not of length
        self.number_of_tuple_elements.
      TypeError: if an element of tuple_shapes cannot be converted to
        a TensorShape.
    "
  [ self tuple_shapes ]
  (py/call-attr self "set_tuple_shapes"  self tuple_shapes ))

(defn set-tuple-types 
  "Sets the type of each element of the queue.

    tuple_types must be a list of length
    self.number_of_tuple_elements, and each element must be
    convertible to a dtype.

    Args:
      tuple_types: the types of each queue element.

    Raises:
      ValueError: if tuple_types is not of length
        self.number_of_tuple_elements.
      TypeError: if an element of tuple_types cannot be converted to a
        dtype.
    "
  [ self tuple_types ]
  (py/call-attr self "set_tuple_types"  self tuple_types ))

(defn shard-dimensions 
  "Gets the shard dimension of each tuple element.

    Returns:
      A list of length number_of_tuple_elements, where each list entry
      is the shard dimension of that tuple element or None if the
      shard dimension has not been set.
    "
  [ self ]
    (py/call-attr self "shard_dimensions"))

(defn sharding-policies 
  "Returns the sharding policies of the InfeedQueue tuple elements."
  [ self ]
    (py/call-attr self "sharding_policies"))

(defn split-inputs-and-generate-enqueue-ops 
  "POORLY-PERFORMING ON MULTI-HOST SYSTEMS.

    Generates the host-side Ops to enqueue a tuple.

    This method performs poorly because it takes an entire input on a single
    host, splits it, and distributes it to all of the cores. It is present only
    to simplify tutorial examples.

    inputs is a list of Tensors to use to feed the queue. Each input is split
    into self.number_of_shards shards. Returns an Op for each shard to enqueue
    the shard. The Op for shard i is placed on device placement_function(i).

    Implicitly freezes the queue configuration if it is not already
    frozen. If the configuration has already been frozen, and is not
    compatible with the types and shapes of inputs, an error
    will be raised.

    Args:
      inputs: a list of Tensors which indicates the types and shapes of the
        queue tuple.
     device_assignment: if not `None`, a TPU `DeviceAssignment`. If
        device_assignment is not `None`, but `placement_function` and
        `ordinal_function` are None, then `device_assignment` will be used to
        place infeeds on the first k TPU shards, where k is the number of shards
        in the queue. If all three are `None`, then default placement and
        ordinal functions are used.
      placement_function: if not None, a function that takes the shard
        index as input and returns a device string indicating which
        device the shard's infeed should be placed on. If placement_function
        and tpu_ordinal_function are None, inputs are sharded round-robin
        across the devices in the system.
      tpu_ordinal_function: if not None, a function that takes the
        shard index as input and returns the ordinal of the TPU device
        the shard's infeed should be placed on. If placement_function
        and tpu_ordinal_function are None, inputs are sharded round-robin
        across the devices in the system.

    Returns:
      A list of host-side Ops, one for each shard, that when executed together
      will enqueue a full-size element of infeed.

    Raises:
      ValueError: if the queue configuration has previously been frozen and the
        shapes of the elements of inputs are not compatible with the frozen
        configuration.
      TypeError: if the queue configuration has previously been frozen and the
        types of the elements of inputs are not compatible with the frozen
        configuration.
    "
  [ self inputs device_assignment placement_function tpu_ordinal_function ]
  (py/call-attr self "split_inputs_and_generate_enqueue_ops"  self inputs device_assignment placement_function tpu_ordinal_function ))

(defn tuple-shapes 
  "Returns the shapes of the InfeedQueue tuple elements."
  [ self ]
    (py/call-attr self "tuple_shapes"))

(defn tuple-types 
  "Returns the types of the InfeedQueue tuple elements."
  [ self ]
    (py/call-attr self "tuple_types"))
