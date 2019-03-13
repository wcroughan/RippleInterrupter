"""
Testing keyboard interrupts
"""
import time
if __name__ == "__main__":
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Keyboard Interrupt received. Exiting!")
