import unittest
import importlib.util
import os
import sys

# Load the GUI module directly since it's not in a package
module_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "DS_Deepresearch_GUI", "DS_Deepresearch_GUI.py")
spec = importlib.util.spec_from_file_location("gui", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
parse_outline_sections = module.parse_outline_sections

class TestParseOutlineSections(unittest.TestCase):
    def test_standard_outline(self):
        outline = """I. Introduction\nA. Background\nB. Methods\nII. Results"""
        expected = ["I. Introduction", "A. Background", "B. Methods", "II. Results"]
        self.assertEqual(parse_outline_sections(outline), expected)

    def test_fallback_nonstandard(self):
        outline = "Intro\nDetails\nConclusion"
        self.assertEqual(parse_outline_sections(outline), ["Intro", "Details", "Conclusion"])

if __name__ == "__main__":
    unittest.main()
