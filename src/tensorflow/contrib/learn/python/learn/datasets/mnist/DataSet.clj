(ns tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet
  "Container class for a dataset (deprecated).

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce mnist (import-module "tensorflow.contrib.learn.python.learn.datasets.mnist"))

(defn DataSet 
  "Container class for a dataset (deprecated).

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
  "
  [images labels & {:keys [fake_data one_hot dtype reshape seed]
                       :or {seed None}} ]
    (py/call-attr-kw mnist "DataSet" [images labels] {:fake_data fake_data :one_hot one_hot :dtype dtype :reshape reshape :seed seed }))

(defn epochs-completed 
  ""
  [ self ]
    (py/call-attr self "epochs_completed"))

(defn images 
  ""
  [ self ]
    (py/call-attr self "images"))

(defn labels 
  ""
  [ self ]
    (py/call-attr self "labels"))
(defn next-batch 
  "Return the next `batch_size` examples from this data set."
  [self batch_size  & {:keys [fake_data shuffle]} ]
    (py/call-attr-kw self "next_batch" [batch_size] {:fake_data fake_data :shuffle shuffle }))

(defn num-examples 
  ""
  [ self ]
    (py/call-attr self "num_examples"))
