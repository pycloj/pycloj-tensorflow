(ns tensorflow.python.keras.api.-v1.keras.utils.HDF5Matrix
  "Representation of HDF5 dataset to be used instead of a Numpy array.

  Example:

  ```python
      x_data = HDF5Matrix('input/file.hdf5', 'data')
      model.predict(x_data)
  ```

  Providing `start` and `end` allows use of a slice of the dataset.

  Optionally, a normalizer function (or lambda) can be given. This will
  be called on every slice of data retrieved.

  Arguments:
      datapath: string, path to a HDF5 file
      dataset: string, name of the HDF5 dataset in the file specified
          in datapath
      start: int, start of desired slice of the specified dataset
      end: int, end of desired slice of the specified dataset
      normalizer: function to be called on data when retrieved

  Returns:
      An array-like HDF5 dataset.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce utils (import-module "tensorflow.python.keras.api._v1.keras.utils"))

(defn HDF5Matrix 
  "Representation of HDF5 dataset to be used instead of a Numpy array.

  Example:

  ```python
      x_data = HDF5Matrix('input/file.hdf5', 'data')
      model.predict(x_data)
  ```

  Providing `start` and `end` allows use of a slice of the dataset.

  Optionally, a normalizer function (or lambda) can be given. This will
  be called on every slice of data retrieved.

  Arguments:
      datapath: string, path to a HDF5 file
      dataset: string, name of the HDF5 dataset in the file specified
          in datapath
      start: int, start of desired slice of the specified dataset
      end: int, end of desired slice of the specified dataset
      normalizer: function to be called on data when retrieved

  Returns:
      An array-like HDF5 dataset.
  "
  [datapath dataset & {:keys [start end normalizer]
                       :or {end None normalizer None}} ]
    (py/call-attr-kw utils "HDF5Matrix" [datapath dataset] {:start start :end end :normalizer normalizer }))

(defn dtype 
  "Gets the datatype of the dataset.

    Returns:
        A numpy dtype string.
    "
  [ self ]
    (py/call-attr self "dtype"))

(defn ndim 
  "Gets the number of dimensions (rank) of the dataset.

    Returns:
        An integer denoting the number of dimensions (rank) of the dataset.
    "
  [ self ]
    (py/call-attr self "ndim"))

(defn shape 
  "Gets a numpy-style shape tuple giving the dataset dimensions.

    Returns:
        A numpy-style shape tuple.
    "
  [ self ]
    (py/call-attr self "shape"))

(defn size 
  "Gets the total dataset size (number of elements).

    Returns:
        An integer denoting the number of elements in the dataset.
    "
  [ self ]
    (py/call-attr self "size"))
