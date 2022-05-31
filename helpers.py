from typing import Callable

def constant_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return initial_value

    return func
