from __future__ import print_function
import sys
import threading
from time import sleep


def quit_function(fn_name):
    """
    Raise an exception if the function takes too long to finish.
    
    Args:
        fn_name (str): The name of the function.
    """
    raise Exception(f"Function {fn_name} takes too long to finish! "
                    "Please avoid duplicate operations and improve program efficiency!")

def exit_after(s=600):
    """
    Decorator to raise an exception if a function takes longer than `s` seconds.
    
    Args:
        s (int, optional): The time limit in seconds. Default is 600 seconds.
    
    Returns:
        function: The decorated function.
    """
    def outer(fn):
        def inner(*args, **kwargs):
            # Container for the function's result or exception
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = fn(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(s)

            if thread.is_alive():
                thread.join()  # Ensure thread is finished before raising the exception
                quit_function(fn.__name__)
            elif exception[0] is not None:
                raise exception[0]  # Re-raise the exception from fn
            else:
                return result[0]

        return inner
    return outer
