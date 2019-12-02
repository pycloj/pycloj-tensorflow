(ns tensorflow.-api.v1.compat.v1.resource-loader
  "Resource management library.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce resource-loader (import-module "tensorflow._api.v1.compat.v1.resource_loader"))

(defn get-data-files-path 
  "Get a direct path to the data files colocated with the script.

  Returns:
    The directory where files specified in data attribute of py_test
    and py_binary are stored.
  "
  [  ]
  (py/call-attr resource-loader "get_data_files_path"  ))

(defn get-path-to-datafile 
  "Get the path to the specified file in the data dependencies.

  The path is relative to tensorflow/

  Args:
    path: a string resource path relative to tensorflow/

  Returns:
    The path to the specified file present in the data attribute of py_test
    or py_binary.

  Raises:
    IOError: If the path is not found, or the resource can't be opened.
  "
  [ path ]
  (py/call-attr resource-loader "get_path_to_datafile"  path ))

(defn get-root-dir-with-all-resources 
  "Get a root directory containing all the data attributes in the build rule.

  Returns:
    The path to the specified file present in the data attribute of py_test
    or py_binary. Falls back to returning the same as get_data_files_path if it
    fails to detect a bazel runfiles directory.
  "
  [  ]
  (py/call-attr resource-loader "get_root_dir_with_all_resources"  ))

(defn load-resource 
  "Load the resource at given path, where path is relative to tensorflow/.

  Args:
    path: a string resource path relative to tensorflow/.

  Returns:
    The contents of that resource.

  Raises:
    IOError: If the path is not found, or the resource can't be opened.
  "
  [ path ]
  (py/call-attr resource-loader "load_resource"  path ))
(defn readahead-file-path 
  "Readahead files not implemented; simply returns given path."
  [path  & {:keys [readahead]} ]
    (py/call-attr-kw resource-loader "readahead_file_path" [path] {:readahead readahead }))
