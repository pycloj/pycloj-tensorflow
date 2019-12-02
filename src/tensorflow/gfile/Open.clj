(ns tensorflow.gfile.Open
  "File I/O wrappers without thread locking.

  Note, that this  is somewhat like builtin Python  file I/O, but
  there are  semantic differences to  make it more  efficient for
  some backing filesystems.  For example, a write  mode file will
  not  be opened  until the  first  write call  (to minimize  RPC
  invocations in network filesystems).
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce gfile (import-module "tensorflow.gfile"))
(defn Open 
  "File I/O wrappers without thread locking.

  Note, that this  is somewhat like builtin Python  file I/O, but
  there are  semantic differences to  make it more  efficient for
  some backing filesystems.  For example, a write  mode file will
  not  be opened  until the  first  write call  (to minimize  RPC
  invocations in network filesystems).
  "
  [name  & {:keys [mode]} ]
    (py/call-attr-kw gfile "Open" [name] {:mode mode }))

(defn close 
  "Closes FileIO. Should be called for the WritableFile to be flushed."
  [ self  ]
  (py/call-attr self "close"  self  ))

(defn flush 
  "Flushes the Writable file.

    This only ensures that the data has made its way out of the process without
    any guarantees on whether it's written to disk. This means that the
    data would survive an application crash but not necessarily an OS crash.
    "
  [ self  ]
  (py/call-attr self "flush"  self  ))

(defn mode 
  "Returns the mode in which the file was opened."
  [ self ]
    (py/call-attr self "mode"))

(defn name 
  "Returns the file name."
  [ self ]
    (py/call-attr self "name"))

(defn next 
  ""
  [ self  ]
  (py/call-attr self "next"  self  ))
(defn read 
  "Returns the contents of a file as a string.

    Starts reading from current position in file.

    Args:
      n: Read 'n' bytes if n != -1. If n = -1, reads to end of file.

    Returns:
      'n' bytes of the file (or whole file) in bytes mode or 'n' bytes of the
      string if in string (regular) mode.
    "
  [self   & {:keys [n]} ]
    (py/call-attr-kw self "read" [] {:n n }))

(defn readline 
  "Reads the next line from the file. Leaves the '\n' at the end."
  [ self  ]
  (py/call-attr self "readline"  self  ))

(defn readlines 
  "Returns all lines from the file in a list."
  [ self  ]
  (py/call-attr self "readlines"  self  ))

(defn seek 
  "Seeks to the offset in the file. (deprecated arguments)

Warning: SOME ARGUMENTS ARE DEPRECATED: `(position)`. They will be removed in a future version.
Instructions for updating:
position is deprecated in favor of the offset argument.

Args:
  offset: The byte count relative to the whence argument.
  whence: Valid values for whence are:
    0: start of the file (default)
    1: relative to the current position of the file
    2: relative to the end of file. offset is usually negative."
  [self offset & {:keys [whence position]
                       :or {position None}} ]
    (py/call-attr-kw self "seek" [offset] {:whence whence :position position }))

(defn seekable 
  "Returns True as FileIO supports random access ops of seek()/tell()"
  [ self  ]
  (py/call-attr self "seekable"  self  ))

(defn size 
  "Returns the size of the file."
  [ self  ]
  (py/call-attr self "size"  self  ))

(defn tell 
  "Returns the current position in the file."
  [ self  ]
  (py/call-attr self "tell"  self  ))

(defn write 
  "Writes file_content to the file. Appends to the end of the file."
  [ self file_content ]
  (py/call-attr self "write"  self file_content ))
