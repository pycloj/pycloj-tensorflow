(ns tensorflow.python.keras.api.-v1.keras.preprocessing.image.NumpyArrayIterator
  "Iterator yielding data from a Numpy array.

  Arguments:
      x: Numpy array of input data or tuple.
          If tuple, the second elements is either
          another numpy array or a list of numpy arrays,
          each of which gets passed
          through as an output without any modifications.
      y: Numpy array of targets data.
      image_data_generator: Instance of `ImageDataGenerator`
          to use for random transformations and normalization.
      batch_size: Integer, size of a batch.
      shuffle: Boolean, whether to shuffle the data between epochs.
      sample_weight: Numpy array of sample weights.
      seed: Random seed for data shuffling.
      data_format: String, one of `channels_first`, `channels_last`.
      save_to_dir: Optional directory where to save the pictures
          being yielded, in a viewable format. This is useful
          for visualizing the random transformations being
          applied, for debugging purposes.
      save_prefix: String prefix to use for saving sample
          images (if `save_to_dir` is set).
      save_format: Format to use for saving sample images
          (if `save_to_dir` is set).
      subset: Subset of data (`\"training\"` or `\"validation\"`) if
          validation_split is set in ImageDataGenerator.
      dtype: Dtype to use for the generated arrays.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce image (import-module "tensorflow.python.keras.api._v1.keras.preprocessing.image"))

(defn NumpyArrayIterator 
  "Iterator yielding data from a Numpy array.

  Arguments:
      x: Numpy array of input data or tuple.
          If tuple, the second elements is either
          another numpy array or a list of numpy arrays,
          each of which gets passed
          through as an output without any modifications.
      y: Numpy array of targets data.
      image_data_generator: Instance of `ImageDataGenerator`
          to use for random transformations and normalization.
      batch_size: Integer, size of a batch.
      shuffle: Boolean, whether to shuffle the data between epochs.
      sample_weight: Numpy array of sample weights.
      seed: Random seed for data shuffling.
      data_format: String, one of `channels_first`, `channels_last`.
      save_to_dir: Optional directory where to save the pictures
          being yielded, in a viewable format. This is useful
          for visualizing the random transformations being
          applied, for debugging purposes.
      save_prefix: String prefix to use for saving sample
          images (if `save_to_dir` is set).
      save_format: Format to use for saving sample images
          (if `save_to_dir` is set).
      subset: Subset of data (`\"training\"` or `\"validation\"`) if
          validation_split is set in ImageDataGenerator.
      dtype: Dtype to use for the generated arrays.
  "
  [x y image_data_generator & {:keys [batch_size shuffle sample_weight seed data_format save_to_dir save_prefix save_format subset dtype]
                       :or {sample_weight None seed None data_format None save_to_dir None subset None dtype None}} ]
    (py/call-attr-kw image "NumpyArrayIterator" [x y image_data_generator] {:batch_size batch_size :shuffle shuffle :sample_weight sample_weight :seed seed :data_format data_format :save_to_dir save_to_dir :save_prefix save_prefix :save_format save_format :subset subset :dtype dtype }))

(defn next 
  "For python 2.x.

        # Returns
            The next batch.
        "
  [ self  ]
  (py/call-attr self "next"  self  ))

(defn on-epoch-end 
  ""
  [ self  ]
  (py/call-attr self "on_epoch_end"  self  ))

(defn reset 
  ""
  [ self  ]
  (py/call-attr self "reset"  self  ))
