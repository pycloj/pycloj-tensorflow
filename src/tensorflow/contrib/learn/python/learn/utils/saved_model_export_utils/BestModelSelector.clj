(ns tensorflow.contrib.learn.python.learn.utils.saved-model-export-utils.BestModelSelector
  "A helper that keeps track of export selection candidates.

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
(defonce saved-model-export-utils (import-module "tensorflow.contrib.learn.python.learn.utils.saved_model_export_utils"))

(defn BestModelSelector 
  "A helper that keeps track of export selection candidates.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
  "
  [ event_file_pattern compare_fn ]
  (py/call-attr saved-model-export-utils "BestModelSelector"  event_file_pattern compare_fn ))

(defn update 
  "Records a given checkpoint and exports if this is the best model.

    Args:
      checkpoint_path: the checkpoint path to export.
      eval_result: a dictionary which is usually generated in evaluation runs.
        By default, eval_results contains 'loss' field.

    Returns:
      A string representing the path to the checkpoint to be exported.
      A dictionary of the same type of eval_result.

    Raises:
      ValueError: if checkpoint path is empty.
      ValueError: if eval_results is None object.
    "
  [ self checkpoint_path eval_result ]
  (py/call-attr self "update"  self checkpoint_path eval_result ))
