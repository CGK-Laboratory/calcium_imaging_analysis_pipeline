from typing import Callable, Dict, Any

rec_functions: Dict[str,Dict[str,Any]] = {}

def register(name: str, message=None,suffix=None) -> None:
    
    if message is None:
        message = f"Running function: {name}..."

    def decorator(func: Callable) -> None:
        assert isinstance(func, Callable)
        rec_functions[name] = {
            'function' : func,
            'message' :message,
            'suffix' : suffix}
    return decorator