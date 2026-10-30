"""
Universal Test Runner for LumiTrack (02_Machine_Learning subfolder)
Can be run directly from 02_Machine_Learning:
    python run_tests.py
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(PROJECT_ROOT, "04_Verification_and_Tests")

print(f"[Test Runner] Running tests from: {TEST_DIR}")
loader = unittest.TestLoader()
suite = loader.discover(start_dir=TEST_DIR, pattern="test_*.py")
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

sys.exit(0 if result.wasSuccessful() else 1)
