import os
from avatar.tools.tool import Tool


class Print2File(Tool):
    """
    A class to write debug strings to a specified file.

    Args:
        debug_print_path (str): The file path to write debug strings.
        size (int): The maximum number of characters to read back from the file.
        **kwargs: Additional arguments.
    """
    
    def __init__(self, debug_print_path: str, size: int = 1024, **kwargs):
        self.active = False
        self.debug_print_path = debug_print_path
        self.size = size
        super().__init__()

    def __call__(self, string: str):
        """
        Write the input string to the specified file if the logger is active.

        Args:
            string (str): The string to be written to the file.
        """
        if self.active:
            try:
                string = str(string)
            except Exception as e:
                raise ValueError(f'Cannot convert {string} to string: {e}')
            with open(self.debug_print_path, 'a+') as f:
                f.write(string + '\n')

    def clean_file(self):
        """
        Remove the debug file if it exists.
        """
        if os.path.exists(self.debug_print_path):
            os.remove(self.debug_print_path)

    def disable(self):
        """
        Disable the logging functionality.
        """
        self.active = False

    def enable(self):
        """
        Enable the logging functionality.
        """
        self.active = True

    def get_written(self) -> str:
        """
        Read the contents of the debug file up to the specified size.

        Returns:
            str: The contents of the debug file.
        """
        if not os.path.exists(self.debug_print_path):
            return ''
        with open(self.debug_print_path, 'r') as f:
            string = f.read(self.size)
        return string

    def __str__(self):
        return 'debug_print(string: str) -> NoneType'

    def __repr__(self):
        return f'This function writes the input string to a file. The written content can be accessed up to {self.size} characters for debugging purposes.'
