class MujocoManipError(Exception):
    """Base class for exceptions in MujocoManipulation."""
    pass

class XMLError(MujocoManipError):
    """Exception raised for errors related to xml."""
    pass

class RuntimeError(MujocoManipError):
    """Exception raised for errors during runtime."""
    pass