# common flags among multiple files go here

from tensorflow.python.platform import flags

flags.DEFINE_string("input", "", """TensorFlow 'GraphDef' file to load.""")
flags.DEFINE_boolean("input_binary", True,
                     """Input graph binary or text.""")
flags.DEFINE_string("output", "", """File to save the output graph to.""")
flags.DEFINE_boolean("output_binary", True,
                     """Output graph binary or text.""")
