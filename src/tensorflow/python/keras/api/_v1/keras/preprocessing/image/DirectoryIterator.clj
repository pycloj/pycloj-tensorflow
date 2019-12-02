(ns tensorflow.python.keras.api.-v1.keras.preprocessing.image.DirectoryIterator
  "Iterator capable of reading images from a directory on disk.

  Arguments:
      directory: Path to the directory to read images from.
          Each subdirectory in this directory will be
          considered to contain images from one class,
          or alternatively you could specify class subdirectories
          via the `classes` argument.
      image_data_generator: Instance of `ImageDataGenerator`
          to use for random transformations and normalization.
      target_size: tuple of integers, dimensions to resize input images to.
      color_mode: One of `\"rgb\"`, `\"rgba\"`, `\"grayscale\"`.
          Color mode to read images.
      classes: Optional list of strings, names of subdirectories
          containing images from each class (e.g. `[\"dogs\", \"cats\"]`).
          It will be computed automatically if not set.
      class_mode: Mode for yielding the targets:
          `\"binary\"`: binary targets (if there are only two classes),
          `\"categorical\"`: categorical targets,
          `\"sparse\"`: integer targets,
          `\"input\"`: targets are images identical to input images (mainly
              used to work with autoencoders),
          `None`: no targets get yielded (only input images are yielded).
      batch_size: Integer, size of a batch.
      shuffle: Boolean, whether to shuffle the data between epochs.
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
      interpolation: Interpolation method used to resample the image if the
          target size is different from that of the loaded image.
          Supported methods are \"nearest\", \"bilinear\", and \"bicubic\".
          If PIL version 1.1.3 or newer is installed, \"lanczos\" is also
          supported. If PIL version 3.4.0 or newer is installed, \"box\" and
          \"hamming\" are also supported. By default, \"nearest\" is used.
      dtype: Dtype to use for generated arrays.
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

(defn DirectoryIterator 
  "Iterator capable of reading images from a directory on disk.

  Arguments:
      directory: Path to the directory to read images from.
          Each subdirectory in this directory will be
          considered to contain images from one class,
          or alternatively you could specify class subdirectories
          via the `classes` argument.
      image_data_generator: Instance of `ImageDataGenerator`
          to use for random transformations and normalization.
      target_size: tuple of integers, dimensions to resize input images to.
      color_mode: One of `\"rgb\"`, `\"rgba\"`, `\"grayscale\"`.
          Color mode to read images.
      classes: Optional list of strings, names of subdirectories
          containing images from each class (e.g. `[\"dogs\", \"cats\"]`).
          It will be computed automatically if not set.
      class_mode: Mode for yielding the targets:
          `\"binary\"`: binary targets (if there are only two classes),
          `\"categorical\"`: categorical targets,
          `\"sparse\"`: integer targets,
          `\"input\"`: targets are images identical to input images (mainly
              used to work with autoencoders),
          `None`: no targets get yielded (only input images are yielded).
      batch_size: Integer, size of a batch.
      shuffle: Boolean, whether to shuffle the data between epochs.
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
      interpolation: Interpolation method used to resample the image if the
          target size is different from that of the loaded image.
          Supported methods are \"nearest\", \"bilinear\", and \"bicubic\".
          If PIL version 1.1.3 or newer is installed, \"lanczos\" is also
          supported. If PIL version 3.4.0 or newer is installed, \"box\" and
          \"hamming\" are also supported. By default, \"nearest\" is used.
      dtype: Dtype to use for generated arrays.
  "
  [directory image_data_generator & {:keys [target_size color_mode classes class_mode batch_size shuffle seed data_format save_to_dir save_prefix save_format follow_links subset interpolation dtype]
                       :or {classes None seed None data_format None save_to_dir None subset None dtype None}} ]
    (py/call-attr-kw image "DirectoryIterator" [directory image_data_generator] {:target_size target_size :color_mode color_mode :classes classes :class_mode class_mode :batch_size batch_size :shuffle shuffle :seed seed :data_format data_format :save_to_dir save_to_dir :save_prefix save_prefix :save_format save_format :follow_links follow_links :subset subset :interpolation interpolation :dtype dtype }))

(defn filepaths 
  ""
  [ self ]
    (py/call-attr self "filepaths"))

(defn labels 
  ""
  [ self ]
    (py/call-attr self "labels"))

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

(defn sample-weight 
  ""
  [ self ]
    (py/call-attr self "sample_weight"))

(defn set-processing-attrs 
  "Sets attributes to use later for processing files into a batch.

        # Arguments
            image_data_generator: Instance of `ImageDataGenerator`
                to use for random transformations and normalization.
            target_size: tuple of integers, dimensions to resize input images to.
            color_mode: One of `\"rgb\"`, `\"rgba\"`, `\"grayscale\"`.
                Color mode to read images.
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
            interpolation: Interpolation method used to resample the image if the
                target size is different from that of the loaded image.
                Supported methods are \"nearest\", \"bilinear\", and \"bicubic\".
                If PIL version 1.1.3 or newer is installed, \"lanczos\" is also
                supported. If PIL version 3.4.0 or newer is installed, \"box\" and
                \"hamming\" are also supported. By default, \"nearest\" is used.
        "
  [ self image_data_generator target_size color_mode data_format save_to_dir save_prefix save_format subset interpolation ]
  (py/call-attr self "set_processing_attrs"  self image_data_generator target_size color_mode data_format save_to_dir save_prefix save_format subset interpolation ))
