"""
Directional movement patterns for beat detection.

This module defines the directional goals that the beat tracker follows.
Each pattern represents a time signature or musical style.

Direction Codes:
    VIBE_UP (2): Look for upward peaks
    VIBE_DOWN (-2): Look for downward troughs
    VIBE_RIGHT (1): Look for rightward peaks
    VIBE_LEFT (-1): Look for leftward troughs

Patterns:
    VIBE_TWOFOUR: [DOWN, UP] - 2/4 time signature
    VIBE_THREEFOUR: [DOWN, RIGHT, UP] - 3/4 time signature
    VIBE_FOURFOUR: [DOWN, LEFT, RIGHT, UP] - 4/4 time signature
"""


class Vibe:
    """Iterator over directional movement patterns for beat tracking.

    Cycles through the pattern indefinitely, returning each goal in sequence.
    """

    def __init__(self, pattern):
        self.pattern = pattern
        self.index = 0

    def next(self):
        """Get next directional goal from pattern.

        Returns:
            int: Direction code (VIBE_UP, VIBE_DOWN, VIBE_LEFT, VIBE_RIGHT)
        """
        current_vibe = self.pattern[self.index]
        self.index = (self.index + 1) % len(self.pattern)
        return current_vibe
VIBE_DOWN=-2
VIBE_UP=2
VIBE_LEFT=-1
VIBE_RIGHT=1
VIBE_TWOFOUR=[VIBE_DOWN, VIBE_UP]
VIBE_THREEFOUR=[VIBE_DOWN, VIBE_RIGHT, VIBE_UP]  
VIBE_FOURFOUR=[VIBE_DOWN, VIBE_LEFT, VIBE_RIGHT, VIBE_UP]
def mirror_vibe(vibe):
    """Mirror a directional code (flip  left/right, NOT up/down).

    Args:
        vibe: Direction code to mirror

    Returns:
        int: Mirrored direction code
    """
    if vibe == VIBE_DOWN:
        return VIBE_DOWN
    elif vibe == VIBE_UP:
        return VIBE_UP
    elif vibe == VIBE_LEFT:
        return VIBE_RIGHT
    elif vibe == VIBE_RIGHT:
        return VIBE_LEFT
    else:
        raise ValueError(f"Invalid vibe value: {vibe}")
def mirror_vibe_pattern(pattern):
    """Mirror all directions in a pattern.

    Args:
        pattern: List of directional codes

    Returns:
        list: New pattern with all directions mirrored
    """
    return [mirror_vibe(v) for v in pattern]