class Tool:
    """
    Abstract base class for defining an agent function call.

    Attributes:
        description (str): A string describing the function of this class.
        func_format (str): A string describing the function call format.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the Tool with an optional kb.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """

        for key, value in kwargs.items():
            setattr(self, key, value)
        self.description = self.__repr__()
        self.func_format = self.__str__()

    def __call__(self, *args, **kwargs):
        """
        Placeholder for the implementation of the function call.

        This method should be overridden by subclasses to provide the actual functionality.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("Subclasses should implement this!")

    def __repr__(self):
        """
        Returns a string describing the function of this class.

        This method should be overridden by subclasses to provide the actual description.
        
        Returns:
            str: Description of the function of this class.
        
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("Subclasses should implement this!")

    def __str__(self):
        """
        Returns a string describing the function call format.

        This method should be overridden by subclasses to provide the actual format.
        
        For example, "search(query: str, attribute: str, num_results: int) -> list"
        
        Returns:
            str: Format of the function call.
        
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("Subclasses should implement this!")
