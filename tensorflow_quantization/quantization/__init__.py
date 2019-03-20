from tensorflow.python.platform import flags

import quantization.clean_control_input
import quantization.graph_to_dot
import quantization.quantize_graph

# common flags among multiple files go here

flags.DEFINE_string("input", "", """TensorFlow 'GraphDef' file to load.""")
flags.DEFINE_boolean("input_binary", True,
                     """Input graph binary or text.""")
flags.DEFINE_string("output", "", """File to save the output graph to.""")
flags.DEFINE_boolean("output_binary", True,
                     """Output graph binary or text.""")
