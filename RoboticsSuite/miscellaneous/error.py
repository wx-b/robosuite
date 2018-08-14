class MujocoManipError(Exception):
    """Base class for exceptions in RoboticsSuite."""

    pass


class XMLError(MujocoManipError):
    """Exception raised for errors related to xml."""

    pass


class SimulationError(MujocoManipError):
    """Exception raised for errors during runtime."""

    pass


class RandomizationError(MujocoManipError):
    """Exception raised for really really bad RNG."""

    pass
