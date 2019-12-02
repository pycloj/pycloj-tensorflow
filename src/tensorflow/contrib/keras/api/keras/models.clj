(ns tensorflow.contrib.keras.api.keras.models
  "Keras models API."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce models (import-module "tensorflow.contrib.keras.api.keras.models"))
(defn load-model 
  "Loads a model saved via `save_model`.

  Arguments:
      filepath: One of the following:
          - String, path to the saved model
          - `h5py.File` object from which to load the model
      custom_objects: Optional dictionary mapping names
          (strings) to custom classes or functions to be
          considered during deserialization.
      compile: Boolean, whether to compile the model
          after loading.

  Returns:
      A Keras model instance. If an optimizer was found
      as part of the saved model, the model is already
      compiled. Otherwise, the model is uncompiled and
      a warning will be displayed. When `compile` is set
      to False, the compilation is omitted without any
      warning.

  Raises:
      ImportError: if loading from an hdf5 file and h5py is not available.
      IOError: In case of an invalid savefile.
  "
  [filepath custom_objects  & {:keys [compile]} ]
    (py/call-attr-kw models "load_model" [filepath custom_objects] {:compile compile }))

(defn model-from-config 
  "Instantiates a Keras model from its config.

  Arguments:
      config: Configuration dictionary.
      custom_objects: Optional dictionary mapping names
          (strings) to custom classes or functions to be
          considered during deserialization.

  Returns:
      A Keras model instance (uncompiled).

  Raises:
      TypeError: if `config` is not a dictionary.
  "
  [ config custom_objects ]
  (py/call-attr models "model_from_config"  config custom_objects ))

(defn model-from-json 
  "Parses a JSON model configuration file and returns a model instance.

  Arguments:
      json_string: JSON string encoding a model configuration.
      custom_objects: Optional dictionary mapping names
          (strings) to custom classes or functions to be
          considered during deserialization.

  Returns:
      A Keras model instance (uncompiled).
  "
  [ json_string custom_objects ]
  (py/call-attr models "model_from_json"  json_string custom_objects ))

(defn model-from-yaml 
  "Parses a yaml model configuration file and returns a model instance.

  Arguments:
      yaml_string: YAML string encoding a model configuration.
      custom_objects: Optional dictionary mapping names
          (strings) to custom classes or functions to be
          considered during deserialization.

  Returns:
      A Keras model instance (uncompiled).

  Raises:
      ImportError: if yaml module is not found.
  "
  [ yaml_string custom_objects ]
  (py/call-attr models "model_from_yaml"  yaml_string custom_objects ))

(defn save-model 
  "Saves a model as a TensorFlow SavedModel or HDF5 file.

  The saved model contains:
      - the model's configuration (topology)
      - the model's weights
      - the model's optimizer's state (if any)

  Thus the saved model can be reinstantiated in
  the exact same state, without any of the code
  used for model definition or training.

  _SavedModel serialization_ (not yet added)

  The SavedModel serialization path uses `tf.saved_model.save` to save the model
  and all trackable objects attached to the model (e.g. layers and variables).
  `@tf.function`-decorated methods are also saved. Additional trackable objects
  and functions are added to the SavedModel to allow the model to be
  loaded back as a Keras Model object.

  Arguments:
      model: Keras model instance to be saved.
      filepath: One of the following:
        - String, path where to save the model
        - `h5py.File` object where to save the model
      overwrite: Whether we should overwrite any existing model at the target
        location, or instead ask the user with a manual prompt.
      include_optimizer: If True, save optimizer's state together.
      save_format: Either 'tf' or 'h5', indicating whether to save the model
        to Tensorflow SavedModel or HDF5. Defaults to 'tf' in TF 2.X, and 'h5'
        in TF 1.X.
      signatures: Signatures to save with the SavedModel. Applicable to the 'tf'
        format only. Please see the `signatures` argument in
        `tf.saved_model.save` for details.

  Raises:
      ImportError: If save format is hdf5, and h5py is not available.
  "
  [model filepath & {:keys [overwrite include_optimizer save_format signatures]
                       :or {save_format None signatures None}} ]
    (py/call-attr-kw models "save_model" [model filepath] {:overwrite overwrite :include_optimizer include_optimizer :save_format save_format :signatures signatures }))
