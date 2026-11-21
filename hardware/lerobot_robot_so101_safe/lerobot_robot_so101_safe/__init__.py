"""Safety layer for the SO-101 follower. Registers the robot type so101_safe_follower."""

from .config import SafeFollowerConfig
from .robot import SafeFollower, SafetyCheckError

__all__ = ["SafeFollower", "SafeFollowerConfig", "SafetyCheckError"]
