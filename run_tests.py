"""
Universal Test Runner for LumiTrack
Can be run from ANY directory (project root or 02_Machine_Learning).
Automatically resolves paths and runs all verification tests.
"""

import os
import sys
import unittest

# Resolve project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) in ["02_Machine_Learning", "03_IoT_Host_Application", "04_Verification_and_Tests"]:
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
else:
    PROJECT_ROOT = CURRENT_DIR

TEST_DIR = os.path.join(PROJECT_ROOT, "04_Verification_and_Tests")

print(f"[Test Runner] Running tests from: {TEST_DIR}")
loader = unittest.TestLoader()
suite = loader.discover(start_dir=TEST_DIR, pattern="test_*.py")
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

sys.exit(0 if result.wasSuccessful() else 1)
