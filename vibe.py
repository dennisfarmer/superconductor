class Vibe:
    def __init__(self, pattern):
        self.pattern = pattern
        self.index = 0

    def next(self):
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
    if vibe == VIBE_DOWN:
        return VIBE_UP
    elif vibe == VIBE_UP:
        return VIBE_DOWN
    elif vibe == VIBE_LEFT:
        return VIBE_RIGHT
    elif vibe == VIBE_RIGHT:
        return VIBE_LEFT
    else:
        raise ValueError(f"Invalid vibe value: {vibe}")
def mirror_vibe_pattern(pattern):
    return [mirror_vibe(v) for v in pattern]