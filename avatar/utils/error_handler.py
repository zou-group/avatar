import re
import traceback


def string_exec_error_handler(err, code):
    """
    Handles execution errors by providing detailed information about the error.

    Args:
        err (Exception): The exception that was raised.
        code (str): The code that caused the exception.

    Returns:
        str: The formatted error information with additional context.
    """
    # Get the detailed traceback information
    error_info = traceback.format_exc()
    
    # Define a pattern to find the line number and function name in the traceback
    pattern = re.compile(r'File "<string>", line (\d+), in (\w+)')
    
    # Find all matches of the pattern in the traceback
    match = pattern.findall(error_info)
    
    # Add the code line causing the error to the traceback information
    for line_number, func_name in match:
        if 'def ' + func_name in code:
            code_line = code.split('\n')[int(line_number) - 1].strip()
            error_info = error_info.replace(
                f'File "<string>", line {line_number}, in {func_name}', 
                f'File "<string>", line {line_number}, in {func_name}\n    {code_line}'
            )
    
    # Remove the rest of the lines after "File "<string>"", but keep the last line
    error_info = error_info.split('File "<string>"')[0] + 'File "<string>"' + error_info.split('File "<string>"')[-1]
    
    return error_info
