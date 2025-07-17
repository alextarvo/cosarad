import numpy as np
from . import util_registration as util_reg

class RegistrationBase(object):
    """Base registration class"""
    def __init__(self, np_pc_template=None, np_pc_target=None):
        """Initializes the registration.

        Args:
        np_pc_template: a template PC, representing a reference frame
        np_pc_target:  a target PC to be registered against the template
        """
        self.np_pc_template = None
        self.np_pc_target = None
        self.np_pc_target_registered = None
        self.accuracy = None
        self.set_template(np_pc_template)
        self.set_target(np_pc_target)

    def set_template(self, np_pc_template):
        """Set a template PC that represents a reference frame"""
        self.np_pc_template = np_pc_template

    def set_target(self, np_pc_target):
        """Set a target PC that represents a reference frame"""
        self.np_pc_target = np_pc_target

    def compute_registration_transform(self):
        """Computes (i.e. learns) the registration to transform the target PC into template frame."""
        raise NotImplementedError

    def get_transform(self):
        """Obtains the learned transformation that transforms target PC into the template frame"""
        raise NotImplementedError
