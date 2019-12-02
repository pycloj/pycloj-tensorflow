(ns tensorflow.contrib.checkpoint
  "Tools for working with object-based checkpoints.

Visualization and inspection:
@@dot_graph_from_checkpoint
@@list_objects
@@object_metadata

Managing dependencies:
@@capture_dependencies
@@Checkpointable
@@CheckpointableBase
@@CheckpointableObjectGraph
@@NoDependency
@@split_dependency

Trackable data structures:
@@List
@@Mapping
@@UniqueNameTracker

Checkpoint management:
@@CheckpointManager

Saving and restoring Python state:
@@NumpyState
@@PythonStateWrapper
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce checkpoint (import-module "tensorflow.contrib.checkpoint"))

(defn capture-dependencies 
  "Capture variables created within this scope as `Template` dependencies.

  Requires that `template.variable_scope` is active.

  This scope is intended as a compatibility measure, allowing a trackable
  object to add dependencies on variables created in a block of code which is
  not aware of object-based saving (and instead uses variable names
  heavily). This is how `Template` objects add dependencies on variables and
  sub-`Template`s. Where possible, use `tf.compat.v1.make_template` directly.

  Args:
    template: The `Template` object to register dependencies with.

  Yields:
    None (when used as a context manager).
  "
  [ template ]
  (py/call-attr checkpoint "capture_dependencies"  template ))

(defn dot-graph-from-checkpoint 
  "Visualizes an object-based checkpoint (from `tf.train.Checkpoint`).

  Useful for inspecting checkpoints and debugging loading issues.

  Example usage from Python (requires pydot):
  ```python
  import tensorflow as tf
  import pydot

  dot_string = tf.contrib.checkpoint.dot_graph_from_checkpoint('/path/to/ckpt')
  parsed, = pydot.graph_from_dot_data(dot_string)
  parsed.write_svg('/tmp/tensorflow/visualized_checkpoint.svg')
  ```

  Example command line usage:
  ```sh
  python -c \"import tensorflow as tf;\
    print(tf.contrib.checkpoint.dot_graph_from_checkpoint('/path/to/ckpt'))\"\
    | dot -Tsvg > /tmp/tensorflow/checkpoint_viz.svg
  ```

  Args:
    save_path: The checkpoint prefix, as returned by `tf.train.Checkpoint.save`
      or `tf.train.latest_checkpoint`.
  Returns:
    A graph in DOT format as a string.
  "
  [ save_path ]
  (py/call-attr checkpoint "dot_graph_from_checkpoint"  save_path ))

(defn list-objects 
  "Traverse the object graph and list all accessible objects.

  Looks for `Trackable` objects which are dependencies of
  `root_trackable`. Includes slot variables only if the variable they are
  slotting for and the optimizer are dependencies of `root_trackable`
  (i.e. if they would be saved with a checkpoint).

  Args:
    root_trackable: A `Trackable` object whose dependencies should be flattened.

  Returns:
    A flat list of objects.
  "
  [ root_trackable ]
  (py/call-attr checkpoint "list_objects"  root_trackable ))

(defn object-metadata 
  "Retrieves information about the objects in a checkpoint.

  Example usage:

  ```python
  object_graph = tf.contrib.checkpoint.object_metadata(
      tf.train.latest_checkpoint(checkpoint_directory))
  ckpt_variable_names = set()
  for node in object_graph.nodes:
    for attribute in node.attributes:
      ckpt_variable_names.add(attribute.full_name)
  ```

  Args:
    save_path: The path to the checkpoint, as returned by `save` or
      `tf.train.latest_checkpoint`.

  Returns:
    A parsed `tf.contrib.checkpoint.TrackableObjectGraph` protocol buffer.
  Raises:
    ValueError: If an object graph was not found in the checkpoint.
  "
  [ save_path ]
  (py/call-attr checkpoint "object_metadata"  save_path ))

(defn split-dependency 
  "Creates multiple dependencies with a synchronized save/restore.

  Useful when a single op produces `Tensor`s which should each be saved under
  different objects, or when `Tensor`s saved with many different objects need to
  be restored together as inputs to a single op (i.e. an object which uses a
  single fused op may be swapped out for a subgraph of objects, and these two
  programs are checkpoint compatible).

  Args:
    component_names: A sequence of names for the split
      dependencies. `fill_save_buffer_fn` must add these keys to the dictionary
      it is passed, and `consume_restore_buffer_fn` will receive a dictionary
      with these keys.
    component_dtypes: Data types for the `Tensor`s being saved and restored, a
      sequence corresponding to `component_names`.
    fill_save_buffer_fn: A function which takes an empty dictionary as an
      argument and adds `Tensor`s with `component_names` as keys. These
      `Tensor`s will be saved as if they were individual variables.
    consume_restore_buffer_fn: A function which takes a dictionary with
      `component_names` as keys mapping to restored individual `Tensor`s and
      returns a restore op (or if executing eagerly, runs the restoration and
      may return `None`).
    device: The device on which to run save and restore operations.

  Returns:
    A dictionary mapping from names to Trackable objects. If one is
    reachable from an object as a dependency, the others should be too; adding
    dependencies on some but not all of the objects will result in errors.
  "
  [ component_names component_dtypes fill_save_buffer_fn consume_restore_buffer_fn device ]
  (py/call-attr checkpoint "split_dependency"  component_names component_dtypes fill_save_buffer_fn consume_restore_buffer_fn device ))
