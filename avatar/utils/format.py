from functools import wraps
from typeguard import typechecked


def format_checked(func):
    """
    Decorator for checking types and non-emptiness of specific argument types for a function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The wrapped function with type and non-emptiness checks.
    """
    checked_func = typechecked(func)
    function_name = func.__name__
    types_to_check = (str, list, dict, tuple, set)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # After type checking, check for non-emptiness
        for idx, arg in enumerate(args):
            arg_name = func.__code__.co_varnames[idx]
            if isinstance(arg, types_to_check) and len(arg) == 0:
                raise ValueError(f"Argument '{arg_name}' in function '{function_name}' has zero length")
        for key, value in kwargs.items():
            if isinstance(value, types_to_check) and len(value) == 0:
                raise ValueError(f"Argument '{key}' in function '{function_name}' is empty")
        return checked_func(*args, **kwargs)

    return wrapper
