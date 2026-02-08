#!/usr/bin/env python3
"""
run_with_delay.py

Simple wrapper that waits before running update_data.py
Helps avoid 403 errors by spacing out requests.

Usage:
    python run_with_delay.py              # Wait 5 minutes then run
    python run_with_delay.py --wait 10    # Wait 10 minutes then run
    python run_with_delay.py --now        # Run immediately
"""

import time
import subprocess
import sys
from datetime import datetime, timedelta


def wait_and_run(wait_minutes=5):
    """Wait for specified minutes, then run update_data.py"""

    if wait_minutes > 0:
        print("=" * 80)
        print("DELAYED START - Avoiding Rate Limits")
        print("=" * 80)
        print(f"\n⏰ Current time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"⏳ Waiting {wait_minutes} minutes before starting...")

        run_time = datetime.now() + timedelta(minutes=wait_minutes)
        print(f"🚀 Will start at: {run_time.strftime('%H:%M:%S')}")
        print("\n💡 Why? FBref blocks rapid requests. Waiting helps avoid 403 errors.")
        print("\nPress Ctrl+C to cancel\n")

        # Countdown
        for remaining in range(wait_minutes * 60, 0, -60):
            mins = remaining // 60
            print(f"   {mins} minute{'s' if mins != 1 else ''} remaining...", flush=True)
            time.sleep(60)

        print("\n✅ Wait complete! Starting now...\n")

    # Run the update script
    print("=" * 80)
    print("Running Data Update")
    print("=" * 80)
    print("")

    try:
        result = subprocess.run(
            [sys.executable, "update_data.py"],
            check=False
        )
        return result.returncode
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run update_data.py with a delay to avoid rate limits"
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=5,
        help="Minutes to wait before running (default: 5)"
    )
    parser.add_argument(
        "--now",
        action="store_true",
        help="Run immediately without waiting"
    )

    args = parser.parse_args()

    if args.now:
        wait_minutes = 0
    else:
        wait_minutes = args.wait

    try:
        return wait_and_run(wait_minutes)
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())