"""Send a graceful stop signal to a Modal training run.

Usage:
    python tools/stop_training.py nafnet_w32_mid4
    python tools/stop_training.py nafnet_w32_mid4 --clear   # remove the stop signal
"""
import modal
import sys

stop_dict = modal.Dict.from_name("train-signals", create_if_missing=True)

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/stop_training.py <checkpoint_dir_name> [--clear]")
        print("  e.g. python tools/stop_training.py nafnet_w32_mid4")
        sys.exit(1)

    key = sys.argv[1]
    clear = "--clear" in sys.argv

    if clear:
        stop_dict[key] = False
        print(f"Cleared stop signal for '{key}'")
    else:
        stop_dict[key] = True
        print(f"Stop signal sent for '{key}'")
        print(f"Training will save checkpoint and exit within ~50 iterations.")


if __name__ == "__main__":
    main()
