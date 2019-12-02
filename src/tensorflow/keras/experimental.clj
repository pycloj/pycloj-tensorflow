(ns tensorflow.python.keras.api.-v1.keras.experimental
  "Public API for tf.keras.experimental namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow.python.keras.api._v1.keras.experimental"))

(defn export-saved-model 
  "Exports a `tf.keras.Model` as a Tensorflow SavedModel. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use `model.save(..., save_format=\"tf\")` or `tf.keras.models.save_model(..., save_format=\"tf\")`.

Note that at this time, subclassed models can only be saved using
`serving_only=True`.

The exported `SavedModel` is a standalone serialization of Tensorflow objects,
and is supported by TF language APIs and the Tensorflow Serving system.
To load the model, use the function
`tf.keras.experimental.load_from_saved_model`.

The `SavedModel` contains:

1. a checkpoint containing the model weights.
2. a `SavedModel` proto containing the Tensorflow backend graph. Separate
   graphs are saved for prediction (serving), train, and evaluation. If
   the model has not been compiled, then only the graph computing predictions
   will be exported.
3. the model's json config. If the model is subclassed, this will only be
   included if the model's `get_config()` method is overwritten.

Example:

```python
import tensorflow as tf

# Create a tf.keras model.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=[10]))
model.summary()

# Save the tf.keras model in the SavedModel format.
path = '/tmp/simple_keras_model'
tf.keras.experimental.export_saved_model(model, path)

# Load the saved keras model back.
new_model = tf.keras.experimental.load_from_saved_model(path)
new_model.summary()
```

Args:
  model: A `tf.keras.Model` to be saved. If the model is subclassed, the flag
    `serving_only` must be set to True.
  saved_model_path: a string specifying the path to the SavedModel directory.
  custom_objects: Optional dictionary mapping string names to custom classes
    or functions (e.g. custom loss functions).
  as_text: bool, `False` by default. Whether to write the `SavedModel` proto
    in text format. Currently unavailable in serving-only mode.
  input_signature: A possibly nested sequence of `tf.TensorSpec` objects, used
    to specify the expected model inputs. See `tf.function` for more details.
  serving_only: bool, `False` by default. When this is true, only the
    prediction graph is saved.

Raises:
  NotImplementedError: If the model is a subclassed model, and serving_only is
    False.
  ValueError: If the input signature cannot be inferred from the model.
  AssertionError: If the SavedModel directory already exists and isn't empty."
  [model saved_model_path custom_objects & {:keys [as_text input_signature serving_only]
                       :or {input_signature None}} ]
    (py/call-attr-kw experimental "export_saved_model" [model saved_model_path custom_objects] {:as_text as_text :input_signature input_signature :serving_only serving_only }))

(defn load-from-saved-model 
  "Loads a keras Model from a SavedModel created by `export_saved_model()`. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
The experimental save and load functions have been  deprecated. Please switch to `tf.keras.models.load_model`.

This function reinstantiates model state by:
1) loading model topology from json (this will eventually come
   from metagraph).
2) loading model weights from checkpoint.

Example:

```python
import tensorflow as tf

# Create a tf.keras model.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=[10]))
model.summary()

# Save the tf.keras model in the SavedModel format.
path = '/tmp/simple_keras_model'
tf.keras.experimental.export_saved_model(model, path)

# Load the saved keras model back.
new_model = tf.keras.experimental.load_from_saved_model(path)
new_model.summary()
```

Args:
  saved_model_path: a string specifying the path to an existing SavedModel.
  custom_objects: Optional dictionary mapping names
      (strings) to custom classes or functions to be
      considered during deserialization.

Returns:
  a keras.Model instance."
  [ saved_model_path custom_objects ]
  (py/call-attr experimental "load_from_saved_model"  saved_model_path custom_objects ))

(defn terminate-keras-multiprocessing-pools 
  "Destroy Keras' multiprocessing pools to prevent deadlocks.

  In general multiprocessing.Pool can interact quite badly with other, seemingly
  unrelated, parts of a codebase due to Pool's reliance on fork. This method
  cleans up all pools which are known to belong to Keras (and thus can be safely
  terminated).

  Args:
    grace_period: Time (in seconds) to wait for process cleanup to propagate.
    use_sigkill: Boolean of whether or not to perform a cleanup pass using
      SIGKILL.

  Returns:
    A list of human readable strings describing all issues encountered. It is up
    to the caller to decide whether to treat this as an error condition.
  "
  [ & {:keys [grace_period use_sigkill]} ]
   (py/call-attr-kw experimental "terminate_keras_multiprocessing_pools" [] {:grace_period grace_period :use_sigkill use_sigkill }))
