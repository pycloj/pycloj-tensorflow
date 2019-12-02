(ns tensorflow.contrib.tensor-forest.client.random-forest
  "A tf.learn implementation of online extremely random forests."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce random-forest (import-module "tensorflow.contrib.tensor_forest.client.random_forest"))

(defn get-combined-model-fn 
  "Get a combined model function given a list of other model fns.

  The model function returned will call the individual model functions and
  combine them appropriately.  For:

  training ops: tf.group them.
  loss: average them.
  predictions: concat probabilities such that predictions[*][0-C1] are the
    probabilities for output 1 (where C1 is the number of classes in output 1),
    predictions[*][C1-(C1+C2)] are the probabilities for output 2 (where C2
    is the number of classes in output 2), etc.  Also stack predictions such
    that predictions[i][j] is the class prediction for example i and output j.

  This assumes that labels are 2-dimensional, with labels[i][j] being the
  label for example i and output j, where forest j is trained using only
  output j.

  Args:
    model_fns: A list of model functions obtained from get_model_fn.

  Returns:
    A ModelFnOps instance.
  "
  [ model_fns ]
  (py/call-attr random-forest "get_combined_model_fn"  model_fns ))

(defn get-model-fn 
  "Return a model function given a way to construct a graph builder."
  [params graph_builder_class device_assigner feature_columns weights_name model_head keys_name & {:keys [early_stopping_rounds early_stopping_loss_threshold num_trainers trainer_id report_feature_importances local_eval head_scope include_all_in_serving output_type]
                       :or {head_scope None}} ]
    (py/call-attr-kw random-forest "get_model_fn" [params graph_builder_class device_assigner feature_columns weights_name model_head keys_name] {:early_stopping_rounds early_stopping_rounds :early_stopping_loss_threshold early_stopping_loss_threshold :num_trainers num_trainers :trainer_id trainer_id :report_feature_importances report_feature_importances :local_eval local_eval :head_scope head_scope :include_all_in_serving include_all_in_serving :output_type output_type }))
