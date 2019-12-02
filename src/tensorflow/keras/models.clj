(ns tensorflow.python.keras.api.-v1.keras.models
  "Code for model cloning, plus model-related API entries.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce models (import-module "tensorflow.python.keras.api._v1.keras.models"))

(defn clone-model 
  "Clone any `Model` instance.

  Model cloning is similar to calling a model on new inputs,
  except that it creates new layers (and thus new weights) instead
  of sharing the weights of the existing layers.

  Arguments:
      model: Instance of `Model`
          (could be a functional model or a Sequential model).
      input_tensors: optional list of input tensors or InputLayer objects
          to build the model upon. If not provided,
          placeholders will be created.
      clone_function: Callable to be used to clone each layer in the target
          model (except `InputLayer` instances). It takes as argument the layer
          instance to be cloned, and returns the corresponding layer instance to
          be used in the model copy. If unspecified, this callable defaults to
          the following serialization/deserialization function:
          `lambda layer: layer.__class__.from_config(layer.get_config())`.
          By passing a custom callable, you can customize your copy of the
          model, e.g. by wrapping certain layers of interest (you might want to
          replace all `LSTM` instances with equivalent
          `Bidirectional(LSTM(...))` instances, for example).

  Returns:
      An instance of `Model` reproducing the behavior
      of the original model, on top of new inputs tensors,
      using newly instantiated weights. The cloned model might behave
      differently from the original model if a custom clone_function
      modifies the layer.

  Raises:
      ValueError: in case of invalid `model` argument value.
  "
  [ model input_tensors clone_function ]
  (py/call-attr models "clone_model"  model input_tensors clone_function ))
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
