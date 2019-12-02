(ns tensorflow.-api.v1.compat.v2.test
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
(defonce test (import-module "tensorflow._api.v1.compat.v2.test"))

(defn assert-equal-graph-def 
  "Asserts that two `GraphDef`s are (mostly) the same.

  Compares two `GraphDef` protos for equality, ignoring versions and ordering of
  nodes, attrs, and control inputs.  Node names are used to match up nodes
  between the graphs, so the naming of nodes must be consistent. This function
  ignores randomized attribute values that may appear in V2 checkpoints.

  Args:
    expected: The `GraphDef` we expected.
    actual: The `GraphDef` we have.

  Raises:
    AssertionError: If the `GraphDef`s do not match.
    TypeError: If either argument is not a `GraphDef`.
  "
  [ expected actual ]
  (py/call-attr test "assert_equal_graph_def"  expected actual ))

(defn benchmark-config 
  "Returns a tf.compat.v1.ConfigProto for disabling the dependency optimizer.

    Returns:
      A TensorFlow ConfigProto object.
  "
  [  ]
  (py/call-attr test "benchmark_config"  ))
(defn compute-gradient 
  "Computes the theoretical and numeric Jacobian of `f`.

  With y = f(x), computes the theoretical and numeric Jacobian dy/dx.

  Args:
    f: the function.
    x: a list arguments for the function
    delta: (optional) perturbation used to compute numeric Jacobian.

  Returns:
    A pair of lists, where the first is a list of 2-d numpy arrays representing
    the theoretical Jacobians for each argument, and the second list is the
    numerical ones. Each 2-d array has \"x_size\" rows
    and \"y_size\" columns where \"x_size\" is the number of elements in the
    corresponding argument and \"y_size\" is the number of elements in f(x).

  Raises:
    ValueError: If result is empty but the gradient is nonzero.
    ValueError: If x is not list, but any other type.

  Example:
  ```python
  @tf.function
  def test_func(x):
    return x*x

  theoretical, numerical = tf.test.compute_gradient(test_func, [1.0])
  theoretical, numerical
  # ((array([[2.]], dtype=float32),), (array([[2.000004]], dtype=float32),))
  ```
  "
  [f x  & {:keys [delta]} ]
    (py/call-attr-kw test "compute_gradient" [f x] {:delta delta }))

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
