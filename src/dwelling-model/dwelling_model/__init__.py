__version__ = "0.1.4"

import os
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
default_input_path = os.path.join(main_path, 'defaults')
default_output_path = os.path.join(main_path, 'outputs')

from .Equipment import *
from .Dwelling import Dwelling
