import os

from robosuite.environments.base import make
from robosuite.environments.sawyer_lift import SawyerLift
from robosuite.environments.sawyer_fit import SawyerFit
from robosuite.environments.sawyer_assembly import SawyerAssembly
from robosuite.environments.sawyer_lego import SawyerLego
from robosuite.environments.sawyer_lego import SawyerLegoEasy
from robosuite.environments.sawyer_lego import SawyerLegoFit
from robosuite.environments.sawyer_stack import SawyerStack
from robosuite.environments.sawyer_pick_place import SawyerPickPlace
from robosuite.environments.sawyer_clutter import SawyerClutter
from robosuite.environments.sawyer_nut_assembly import SawyerNutAssembly

from robosuite.environments.baxter_lift import BaxterLift
from robosuite.environments.baxter_peg_in_hole import BaxterPegInHole

__version__ = "0.1.0"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
