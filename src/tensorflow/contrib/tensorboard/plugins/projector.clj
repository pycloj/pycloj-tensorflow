(ns tensorflow.contrib.tensorboard.plugins.projector
  "Public API for the Embedding Projector.

@@ProjectorPluginAsset
@@ProjectorConfig
@@EmbeddingInfo
@@EmbeddingMetadata
@@SpriteMetadata
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce projector (import-module "tensorflow.contrib.tensorboard.plugins.projector"))

(defn visualize-embeddings 
  "Stores a config file used by the embedding projector.

  Args:
    summary_writer: The summary writer used for writing events.
    config: `tf.contrib.tensorboard.plugins.projector.ProjectorConfig`
      proto that holds the configuration for the projector such as paths to
      checkpoint files and metadata files for the embeddings. If
      `config.model_checkpoint_path` is none, it defaults to the
      `logdir` used by the summary_writer.

  Raises:
    ValueError: If the summary writer does not have a `logdir`.
  "
  [ summary_writer config ]
  (py/call-attr projector "visualize_embeddings"  summary_writer config ))
