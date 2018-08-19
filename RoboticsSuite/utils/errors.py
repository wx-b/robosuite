class RoboticsSuiteError(Exception):
    """Base class for exceptions in RoboticsSuite."""
    pass


class XMLError(RoboticsSuiteError):
    """Exception raised for errors related to xml."""
    pass


class SimulationError(RoboticsSuiteError):
    """Exception raised for errors during runtime."""
    pass


class RandomizationError(RoboticsSuiteError):
    """Exception raised for really really bad RNG."""
    pass
