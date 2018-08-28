import os

from RoboticsSuite.environments.base import make
from RoboticsSuite.environments.sawyer_lift import SawyerLift
from RoboticsSuite.environments.sawyer_stack import SawyerStack
from RoboticsSuite.environments.sawyer_pick_place import SawyerPickPlace
from RoboticsSuite.environments.sawyer_nut_assembly import SawyerNutAssembly

from RoboticsSuite.environments.baxter_lift import BaxterLift
from RoboticsSuite.environments.baxter_peg_in_hole import BaxterPegInHole
from RoboticsSuite.wrappers import DataCollectionWrapper

__version__ = "0.1.0"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""

assets_path = os.path.join(__path__[0], "models/assets")
