(ns tensorflow.-api.v1.compat.v1.test
  "Testing.

See the [Testing](https://tensorflow.org/api_docs/python/tf/test) guide.

Note: `tf.compat.v1.test.mock` is an alias to the python `mock` or
`unittest.mock` depending on the python version.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce test (import-module "tensorflow._api.v1.compat.v1.test"))
(defn assert-equal-graph-def 
  "Asserts that two `GraphDef`s are (mostly) the same.

  Compares two `GraphDef` protos for equality, ignoring versions and ordering of
  nodes, attrs, and control inputs.  Node names are used to match up nodes
  between the graphs, so the naming of nodes must be consistent.

  Args:
    actual: The `GraphDef` we have.
    expected: The `GraphDef` we expected.
    checkpoint_v2: boolean determining whether to ignore randomized attribute
      values that appear in V2 checkpoints.
    hash_table_shared_name: boolean determining whether to ignore randomized
      shared_names that appear in HashTableV2 op defs.

  Raises:
    AssertionError: If the `GraphDef`s do not match.
    TypeError: If either argument is not a `GraphDef`.
  "
  [actual expected  & {:keys [checkpoint_v2 hash_table_shared_name]} ]
    (py/call-attr-kw test "assert_equal_graph_def" [actual expected] {:checkpoint_v2 checkpoint_v2 :hash_table_shared_name hash_table_shared_name }))

(defn benchmark-config 
  "Returns a tf.compat.v1.ConfigProto for disabling the dependency optimizer.

    Returns:
      A TensorFlow ConfigProto object.
  "
  [  ]
  (py/call-attr test "benchmark_config"  ))

(defn compute-gradient 
  "Computes and returns the theoretical and numerical Jacobian. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.test.compute_gradient in 2.0, which has better support for functions. Note that the two versions have different usage, so code change is needed.

If `x` or `y` is complex, the Jacobian will still be real but the
corresponding Jacobian dimension(s) will be twice as large.  This is required
even if both input and output is complex since TensorFlow graphs are not
necessarily holomorphic, and may have gradients not expressible as complex
numbers.  For example, if `x` is complex with shape `[m]` and `y` is complex
with shape `[n]`, each Jacobian `J` will have shape `[m * 2, n * 2]` with

    J[:m, :n] = d(Re y)/d(Re x)
    J[:m, n:] = d(Im y)/d(Re x)
    J[m:, :n] = d(Re y)/d(Im x)
    J[m:, n:] = d(Im y)/d(Im x)

Args:
  x: a tensor or list of tensors
  x_shape: the dimensions of x as a tuple or an array of ints. If x is a list,
  then this is the list of shapes.
  y: a tensor
  y_shape: the dimensions of y as a tuple or an array of ints.
  x_init_value: (optional) a numpy array of the same shape as \"x\"
    representing the initial value of x. If x is a list, this should be a list
    of numpy arrays.  If this is none, the function will pick a random tensor
    as the initial value.
  delta: (optional) the amount of perturbation.
  init_targets: list of targets to run to initialize model params.
  extra_feed_dict: dict that allows fixing specified tensor values
    during the Jacobian calculation.

Returns:
  Two 2-d numpy arrays representing the theoretical and numerical
  Jacobian for dy/dx. Each has \"x_size\" rows and \"y_size\" columns
  where \"x_size\" is the number of elements in x and \"y_size\" is the
  number of elements in y. If x is a list, returns a list of two numpy arrays."
  [x x_shape y y_shape x_init_value & {:keys [delta init_targets extra_feed_dict]
                       :or {init_targets None extra_feed_dict None}} ]
    (py/call-attr-kw test "compute_gradient" [x x_shape y y_shape x_init_value] {:delta delta :init_targets init_targets :extra_feed_dict extra_feed_dict }))

(defn compute-gradient-error 
  "Computes the gradient error. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.test.compute_gradient in 2.0, which has better support for functions. Note that the two versions have different usage, so code change is needed.

Computes the maximum error for dy/dx between the computed Jacobian and the
numerically estimated Jacobian.

This function will modify the tensors passed in as it adds more operations
and hence changing the consumers of the operations of the input tensors.

This function adds operations to the current session. To compute the error
using a particular device, such as a GPU, use the standard methods for
setting a device (e.g. using with sess.graph.device() or setting a device
function in the session constructor).

Args:
  x: a tensor or list of tensors
  x_shape: the dimensions of x as a tuple or an array of ints. If x is a list,
  then this is the list of shapes.
  y: a tensor
  y_shape: the dimensions of y as a tuple or an array of ints.
  x_init_value: (optional) a numpy array of the same shape as \"x\"
    representing the initial value of x. If x is a list, this should be a list
    of numpy arrays.  If this is none, the function will pick a random tensor
    as the initial value.
  delta: (optional) the amount of perturbation.
  init_targets: list of targets to run to initialize model params.
  extra_feed_dict: dict that allows fixing specified tensor values
    during the Jacobian calculation.

Returns:
  The maximum error in between the two Jacobians."
  [x x_shape y y_shape x_init_value & {:keys [delta init_targets extra_feed_dict]
                       :or {init_targets None extra_feed_dict None}} ]
    (py/call-attr-kw test "compute_gradient_error" [x x_shape y y_shape x_init_value] {:delta delta :init_targets init_targets :extra_feed_dict extra_feed_dict }))

(defn create-local-cluster 
  "Create and start local servers and return the associated `Server` objects.

  \"PS\" stands for \"parameter server\": a task responsible for storing and
  updating the model's parameters. Other tasks send updates to these parameters
  as they work on optimizing the parameters. This particular division of labor
  between tasks is not required, but is common for distributed training.

  Read more at https://www.tensorflow.org/guide/extend/architecture

  ![components](https://www.tensorflow.org/images/diag1.svg \"components\")


  Figure illustrates the interaction of these components.
  \"/job:worker/task:0\" and \"/job:ps/task:0\" are both tasks with worker services.


  Example:
  ```python
  workers, _ = tf.test.create_local_cluster(num_workers=2, num_ps=2)

  worker_sessions = [tf.compat.v1.Session(w.target) for w in workers]

  with tf.device(\"/job:ps/task:0\"):
    ...
  with tf.device(\"/job:ps/task:1\"):
    ...
  with tf.device(\"/job:worker/task:0\"):
    ...
  with tf.device(\"/job:worker/task:1\"):
    ...

  worker_sessions[0].run(...)
  ```

  Args:
    num_workers: Number of worker servers to start.
    num_ps: Number of PS servers to start.
    protocol: Communication protocol. Allowed values are documented in the
      documentation of `tf.distribute.Server`.
    worker_config: (optional) `tf.ConfigProto` to initialize workers. Can be
      used to instantiate multiple devices etc.
    ps_config: (optional) `tf.ConfigProto` to initialize PS servers.

  Returns:
    A tuple `(worker_servers, ps_servers)`.  `worker_servers` is a list
    of `num_workers` objects of type `tf.distribute.Server` (all running
    locally);
    and `ps_servers` is a list of `num_ps` objects of similar type.

  Raises:
    ImportError: if portpicker module was not found at load time
  "
  [num_workers num_ps & {:keys [protocol worker_config ps_config]
                       :or {worker_config None ps_config None}} ]
    (py/call-attr-kw test "create_local_cluster" [num_workers num_ps] {:protocol protocol :worker_config worker_config :ps_config ps_config }))

(defn get-temp-dir 
  "Returns a temporary directory for use during tests.

  There is no need to delete the directory after the test.

  Returns:
    The temporary directory.
  "
  [  ]
  (py/call-attr test "get_temp_dir"  ))

(defn gpu-device-name 
  "Returns the name of a GPU device if available or the empty string."
  [  ]
  (py/call-attr test "gpu_device_name"  ))

(defn is-built-with-cuda 
  "Returns whether TensorFlow was built with CUDA (GPU) support."
  [  ]
  (py/call-attr test "is_built_with_cuda"  ))

(defn is-built-with-gpu-support 
  "Returns whether TensorFlow was built with GPU (i.e. CUDA or ROCm) support."
  [  ]
  (py/call-attr test "is_built_with_gpu_support"  ))

(defn is-built-with-rocm 
  "Returns whether TensorFlow was built with ROCm (GPU) support."
  [  ]
  (py/call-attr test "is_built_with_rocm"  ))

(defn is-gpu-available 
  "Returns whether TensorFlow can access a GPU.

  Warning: if a non-GPU version of the package is installed, the function would
  also return False. Use `tf.test.is_built_with_cuda` to validate if TensorFlow
  was build with CUDA support.

  Args:
    cuda_only: limit the search to CUDA GPUs.
    min_cuda_compute_capability: a (major,minor) pair that indicates the minimum
      CUDA compute capability required, or None if no requirement.

  Note that the keyword arg name \"cuda_only\" is misleading (since routine will
  return true when a GPU device is available irrespective of whether TF was
  built with CUDA support or ROCm support. However no changes here because

  ++ Changing the name \"cuda_only\" to something more generic would break
     backward compatibility

  ++ Adding an equivalent \"rocm_only\" would require the implementation check
     the build type. This in turn would require doing the same for CUDA and thus
     potentially break backward compatibility

  ++ Adding a new \"cuda_or_rocm_only\" would not break backward compatibility,
     but would require most (if not all) callers to update the call to use
     \"cuda_or_rocm_only\" instead of \"cuda_only\"

  Returns:
    True if a GPU device of the requested kind is available.
  "
  [ & {:keys [cuda_only min_cuda_compute_capability]
       :or {min_cuda_compute_capability None}} ]
  
   (py/call-attr-kw test "is_gpu_available" [] {:cuda_only cuda_only :min_cuda_compute_capability min_cuda_compute_capability }))

(defn main 
  "Runs all unit tests."
  [ argv ]
  (py/call-attr test "main"  argv ))

(defn test-src-dir-path 
  "Creates an absolute test srcdir path given a relative path.

  Args:
    relative_path: a path relative to tensorflow root.
      e.g. \"core/platform\".

  Returns:
    An absolute path to the linked in runfiles.
  "
  [ relative_path ]
  (py/call-attr test "test_src_dir_path"  relative_path ))
