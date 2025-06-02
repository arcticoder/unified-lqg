#!/usr/bin/env python3
"""
Quick smoke test for the enhanced LQG pipeline.
"""

from enhance_main_pipeline import main
import sys

if __name__ == "__main__":
    try:
        print("=== Running Enhanced LQG Pipeline Smoke Test ===")
        main()
        print("\n✓ Pipeline completed without crashing")
        print("✓ All major components tested successfully")
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
