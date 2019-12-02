(ns tensorflow.DeviceSpec
  "Represents a (possibly partial) specification for a TensorFlow device.

  `DeviceSpec`s are used throughout TensorFlow to describe where state is stored
  and computations occur. Using `DeviceSpec` allows you to parse device spec
  strings to verify their validity, merge them or compose them programmatically.

  Example:

  ```python
  # Place the operations on device \"GPU:0\" in the \"ps\" job.
  device_spec = DeviceSpec(job=\"ps\", device_type=\"GPU\", device_index=0)
  with tf.device(device_spec):
    # Both my_var and squared_var will be placed on /job:ps/device:GPU:0.
    my_var = tf.Variable(..., name=\"my_variable\")
    squared_var = tf.square(my_var)
  ```

  If a `DeviceSpec` is partially specified, it will be merged with other
  `DeviceSpec`s according to the scope in which it is defined. `DeviceSpec`
  components defined in inner scopes take precedence over those defined in
  outer scopes.

  ```python
  with tf.device(DeviceSpec(job=\"train\", )):
    with tf.device(DeviceSpec(job=\"ps\", device_type=\"GPU\", device_index=0):
      # Nodes created here will be assigned to /job:ps/device:GPU:0.
    with tf.device(DeviceSpec(device_type=\"GPU\", device_index=1):
      # Nodes created here will be assigned to /job:train/device:GPU:1.
  ```

  A `DeviceSpec` consists of 5 components -- each of
  which is optionally specified:

  * Job: The job name.
  * Replica: The replica index.
  * Task: The task index.
  * Device type: The device type string (e.g. \"CPU\" or \"GPU\").
  * Device index: The device index.
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

(defn DeviceSpec 
  "Represents a (possibly partial) specification for a TensorFlow device.

  `DeviceSpec`s are used throughout TensorFlow to describe where state is stored
  and computations occur. Using `DeviceSpec` allows you to parse device spec
  strings to verify their validity, merge them or compose them programmatically.

  Example:

  ```python
  # Place the operations on device \"GPU:0\" in the \"ps\" job.
  device_spec = DeviceSpec(job=\"ps\", device_type=\"GPU\", device_index=0)
  with tf.device(device_spec):
    # Both my_var and squared_var will be placed on /job:ps/device:GPU:0.
    my_var = tf.Variable(..., name=\"my_variable\")
    squared_var = tf.square(my_var)
  ```

  If a `DeviceSpec` is partially specified, it will be merged with other
  `DeviceSpec`s according to the scope in which it is defined. `DeviceSpec`
  components defined in inner scopes take precedence over those defined in
  outer scopes.

  ```python
  with tf.device(DeviceSpec(job=\"train\", )):
    with tf.device(DeviceSpec(job=\"ps\", device_type=\"GPU\", device_index=0):
      # Nodes created here will be assigned to /job:ps/device:GPU:0.
    with tf.device(DeviceSpec(device_type=\"GPU\", device_index=1):
      # Nodes created here will be assigned to /job:train/device:GPU:1.
  ```

  A `DeviceSpec` consists of 5 components -- each of
  which is optionally specified:

  * Job: The job name.
  * Replica: The replica index.
  * Task: The task index.
  * Device type: The device type string (e.g. \"CPU\" or \"GPU\").
  * Device index: The device index.
  "
  [ job replica task device_type device_index ]
  (py/call-attr tensorflow "DeviceSpec"  job replica task device_type device_index ))

(defn device-index 
  ""
  [ self ]
    (py/call-attr self "device_index"))

(defn device-type 
  ""
  [ self ]
    (py/call-attr self "device_type"))

(defn job 
  ""
  [ self ]
    (py/call-attr self "job"))

(defn make-merged-spec 
  "Returns a new DeviceSpec which incorporates `dev`.

    When combining specs, `dev` will take precidence over the current spec.
    So for instance:
    ```
    first_spec = tf.DeviceSpec(job=0, device_type=\"CPU\")
    second_spec = tf.DeviceSpec(device_type=\"GPU\")
    combined_spec = first_spec.make_merged_spec(second_spec)
    ```

    is equivalent to:
    ```
    combined_spec = tf.DeviceSpec(job=0, device_type=\"GPU\")
    ```

    Args:
      dev: a `DeviceSpec`

    Returns:
      A new `DeviceSpec` which combines `self` and `dev`
    "
  [ self dev ]
  (py/call-attr self "make_merged_spec"  self dev ))

(defn merge-from 
  "Merge the properties of \"dev\" into this `DeviceSpec`.

    Note: Will be removed in TensorFlow 2.x since DeviceSpecs will become
          immutable.

    Args:
      dev: a `DeviceSpec`.
    "
  [ self dev ]
  (py/call-attr self "merge_from"  self dev ))

(defn parse-from-string 
  "Parse a `DeviceSpec` name into its components.

    2.x behavior change:
      In TensorFlow 1.x, this function mutates its own state and returns itself.
      In 2.x, DeviceSpecs are immutable, and this function will return a
        DeviceSpec which contains the spec.

      Recommended:
        ```
        # my_spec and my_updated_spec are unrelated.
        my_spec = tf.DeviceSpec.from_string(\"/CPU:0\")
        my_updated_spec = tf.DeviceSpec.from_string(\"/GPU:0\")
        with tf.device(my_updated_spec):
          ...
        ```

      Will work in 1.x and 2.x (though deprecated in 2.x):
        ```
        my_spec = tf.DeviceSpec.from_string(\"/CPU:0\")
        my_updated_spec = my_spec.parse_from_string(\"/GPU:0\")
        with tf.device(my_updated_spec):
          ...
        ```

      Will NOT work in 2.x:
        ```
        my_spec = tf.DeviceSpec.from_string(\"/CPU:0\")
        my_spec.parse_from_string(\"/GPU:0\")  # <== Will not update my_spec
        with tf.device(my_spec):
          ...
        ```

      In general, `DeviceSpec.from_string` should completely replace
      `DeviceSpec.parse_from_string`, and `DeviceSpec.replace` should
      completely replace setting attributes directly.

    Args:
      spec: an optional string of the form
       /job:<name>/replica:<id>/task:<id>/device:CPU:<id>
      or
       /job:<name>/replica:<id>/task:<id>/device:GPU:<id>
      as cpu and gpu are mutually exclusive.
      All entries are optional.

    Returns:
      The `DeviceSpec`.

    Raises:
      ValueError: if the spec was not valid.
    "
  [ self spec ]
  (py/call-attr self "parse_from_string"  self spec ))

(defn replace 
  "Convenience method for making a new DeviceSpec by overriding fields.

    For instance:
    ```
    my_spec = DeviceSpec=(job=\"my_job\", device=\"CPU\")
    my_updated_spec = my_spec.replace(device=\"GPU\")
    my_other_spec = my_spec.replace(device=None)
    ```

    Args:
      **kwargs: This method takes the same args as the DeviceSpec constructor

    Returns:
      A DeviceSpec with the fields specified in kwargs overridden.
    "
  [ self  ]
  (py/call-attr self "replace"  self  ))

(defn replica 
  ""
  [ self ]
    (py/call-attr self "replica"))

(defn task 
  ""
  [ self ]
    (py/call-attr self "task"))

(defn to-string 
  "Return a string representation of this `DeviceSpec`.

    Returns:
      a string of the form
      /job:<name>/replica:<id>/task:<id>/device:<device_type>:<id>.
    "
  [ self  ]
  (py/call-attr self "to_string"  self  ))
