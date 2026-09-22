#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2.2", "matplotlib>=3.8", "openai>=2.0"]
# ///
"""Analyze complete candidate solutions without requiring parent mappings."""

import sys

from compare_vendi import main as compare_main


# 默认比较完整源码方案，复用统一的统计、审计与绘图流程。
def main(argv=None):
    return compare_main(argv, default_mode="solutions")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, TypeError) as exc:
        print(f"compare_solution_vendi: {exc}", file=sys.stderr)
        raise SystemExit(2)
