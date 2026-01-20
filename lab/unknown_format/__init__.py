"""Unknown format lab

Generates families of synthetic binary formats with ground truth and auto-tests.
This is the closed-world training and evaluation environment for the CCP party trick.
"""

from lab.unknown_format.dataset import UnknownFormatLabDataset
from lab.unknown_format.format_family import FormatFamily
from lab.unknown_format.oracle import FormatOracle
from lab.unknown_format.test_gen import ToolTestGenerator

__all__ = [
    "FormatFamily",
    "FormatOracle",
    "ToolTestGenerator",
    "UnknownFormatLabDataset",
]

