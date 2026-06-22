__all__ = ["main", "JudgeApp"]

def __getattr__(name: str):
    if name == "main":
        from .main import main as _main
        return _main
    if name == "JudgeApp":
        from .main import JudgeApp as _JudgeApp
        return _JudgeApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
