"""CPU (gym.Wrapper) environment wrappers: wind (wind.py/wind_wrapper.py) and
procedural obstacles (obstacles.py).

Domain randomization and curriculum learning used to live here too
(DomainRandomizationWrapper/DomainRandomizationConfig, curriculum.py's
CurriculumWrapper) but have been removed: that logic is now handled
GPU-side, in mjc_dronetests — see envspecs/dynamics.py's DomainRandConfig
(physics-parameter randomization, the domain_rand_fn hook on
MJXVectorAviary/MultiVectorAviary) and train.py's reward-curriculum /
reset-function-mixing (--reset a b --reset-weight) for the curriculum
equivalent. Wind and obstacles remain CPU-only for now — no GPU port exists
yet for those.
"""
