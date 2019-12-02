(ns tensorflow.contrib.keras.api.keras.utils
  "Keras utilities."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce utils (import-module "tensorflow.contrib.keras.api.keras.utils"))

(defn convert-all-kernels-in-model 
  "Converts all convolution kernels in a model from Theano to TensorFlow.

  Also works from TensorFlow to Theano.

  Arguments:
      model: target model for the conversion.
  "
  [ model ]
  (py/call-attr utils "convert_all_kernels_in_model"  model ))

(defn custom-object-scope 
  "Provides a scope that changes to `_GLOBAL_CUSTOM_OBJECTS` cannot escape.

  Convenience wrapper for `CustomObjectScope`.
  Code within a `with` statement will be able to access custom objects
  by name. Changes to global custom objects persist
  within the enclosing `with` statement. At end of the `with` statement,
  global custom objects are reverted to state
  at beginning of the `with` statement.

  Example:

  Consider a custom object `MyObject`

  ```python
      with custom_object_scope({'MyObject':MyObject}):
          layer = Dense(..., kernel_regularizer='MyObject')
          # save, load, etc. will recognize custom object by name
  ```

  Arguments:
      *args: Variable length list of dictionaries of name,
          class pairs to add to custom objects.

  Returns:
      Object of type `CustomObjectScope`.
  "
  [  ]
  (py/call-attr utils "custom_object_scope"  ))
(defn deserialize-keras-object 
  ""
  [identifier module_objects custom_objects  & {:keys [printable_module_name]} ]
    (py/call-attr-kw utils "deserialize_keras_object" [identifier module_objects custom_objects] {:printable_module_name printable_module_name }))

(defn get-custom-objects 
  "Retrieves a live reference to the global dictionary of custom objects.

  Updating and clearing custom objects using `custom_object_scope`
  is preferred, but `get_custom_objects` can
  be used to directly access `_GLOBAL_CUSTOM_OBJECTS`.

  Example:

  ```python
      get_custom_objects().clear()
      get_custom_objects()['MyObject'] = MyObject
  ```

  Returns:
      Global dictionary of names to classes (`_GLOBAL_CUSTOM_OBJECTS`).
  "
  [  ]
  (py/call-attr utils "get_custom_objects"  ))

(defn get-file 
  "Downloads a file from a URL if it not already in the cache.

  By default the file at the url `origin` is downloaded to the
  cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
  and given the filename `fname`. The final location of a file
  `example.txt` would therefore be `~/.keras/datasets/example.txt`.

  Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
  Passing a hash will verify the file after download. The command line
  programs `shasum` and `sha256sum` can compute the hash.

  Arguments:
      fname: Name of the file. If an absolute path `/path/to/file.txt` is
          specified the file will be saved at that location.
      origin: Original URL of the file.
      untar: Deprecated in favor of 'extract'.
          boolean, whether the file should be decompressed
      md5_hash: Deprecated in favor of 'file_hash'.
          md5 hash of the file for verification
      file_hash: The expected hash string of the file after download.
          The sha256 and md5 hash algorithms are both supported.
      cache_subdir: Subdirectory under the Keras cache dir where the file is
          saved. If an absolute path `/path/to/folder` is
          specified the file will be saved at that location.
      hash_algorithm: Select the hash algorithm to verify the file.
          options are 'md5', 'sha256', and 'auto'.
          The default 'auto' detects the hash algorithm in use.
      extract: True tries extracting the file as an Archive, like tar or zip.
      archive_format: Archive format to try for extracting the file.
          Options are 'auto', 'tar', 'zip', and None.
          'tar' includes tar, tar.gz, and tar.bz files.
          The default 'auto' is ['tar', 'zip'].
          None or an empty list will return no matches found.
      cache_dir: Location to store cached files, when None it
          defaults to the [Keras
            Directory](/faq/#where-is-the-keras-configuration-filed-stored).

  Returns:
      Path to the downloaded file
  "
  [fname origin & {:keys [untar md5_hash file_hash cache_subdir hash_algorithm extract archive_format cache_dir]
                       :or {md5_hash None file_hash None cache_dir None}} ]
    (py/call-attr-kw utils "get_file" [fname origin] {:untar untar :md5_hash md5_hash :file_hash file_hash :cache_subdir cache_subdir :hash_algorithm hash_algorithm :extract extract :archive_format archive_format :cache_dir cache_dir }))
(defn normalize 
  "Normalizes a Numpy array.

  Arguments:
      x: Numpy array to normalize.
      axis: axis along which to normalize.
      order: Normalization order (e.g. 2 for L2 norm).

  Returns:
      A normalized copy of the array.
  "
  [x  & {:keys [axis order]} ]
    (py/call-attr-kw utils "normalize" [x] {:axis axis :order order }))
(defn plot-model 
  "Converts a Keras model to dot format and save to a file.

  Arguments:
    model: A Keras model instance
    to_file: File name of the plot image.
    show_shapes: whether to display shape information.
    show_layer_names: whether to display layer names.
    rankdir: `rankdir` argument passed to PyDot,
        a string specifying the format of the plot:
        'TB' creates a vertical plot;
        'LR' creates a horizontal plot.
    expand_nested: Whether to expand nested models into clusters.
    dpi: Dots per inch.

  Returns:
    A Jupyter notebook Image object if Jupyter is installed.
    This enables in-line display of the model plots in notebooks.
  "
  [model  & {:keys [to_file show_shapes show_layer_names rankdir expand_nested dpi]} ]
    (py/call-attr-kw utils "plot_model" [model] {:to_file to_file :show_shapes show_shapes :show_layer_names show_layer_names :rankdir rankdir :expand_nested expand_nested :dpi dpi }))

(defn serialize-keras-object 
  ""
  [ instance ]
  (py/call-attr utils "serialize_keras_object"  instance ))
(defn to-categorical 
  "Converts a class vector (integers) to binary class matrix.

  E.g. for use with categorical_crossentropy.

  Arguments:
      y: class vector to be converted into a matrix
          (integers from 0 to num_classes).
      num_classes: total number of classes.
      dtype: The data type expected by the input. Default: `'float32'`.

  Returns:
      A binary matrix representation of the input. The classes axis is placed
      last.
  "
  [y num_classes  & {:keys [dtype]} ]
    (py/call-attr-kw utils "to_categorical" [y num_classes] {:dtype dtype }))
