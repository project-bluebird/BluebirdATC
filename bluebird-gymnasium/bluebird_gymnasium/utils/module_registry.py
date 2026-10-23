"""A class for dynamically/lazily loading modules"""

from collections.abc import Callable


class ModuleRegistry:
    """Registry that loads modules on demand using lazy loading."""

    def __init__(self):
        self._registry = {}
        self._loaded = {}

    def register(self, name: str, module_path: str, factory: Callable | None = None):
        """
        Register a module for lazy loading.

        Args:
            name: Key to access the module
            module_path: Import path like 'package.module:ClassName'
            factory: Optional callable to create instance (default: import the object)
        """
        self._registry[name] = {"path": module_path, "factory": factory}

    def get(self, name: str):  # noqa: ANN201
        """Load and return the module/object if not already loaded."""
        if name in self._loaded:
            return self._loaded[name]

        if name not in self._registry:
            raise KeyError(f"Module '{name}' not registered")

        config = self._registry[name]

        # Load the module using factory if provided, otherwise import from path
        obj = config["factory"]() if config["factory"] else self._import_from_path(config["path"])

        self._loaded[name] = obj
        return obj

    def _import_from_path(self, path: str):  # noqa: ANN202
        """Import object from a string path like 'package.module:ClassName'."""
        if ":" in path:
            module_path, obj_name = path.rsplit(":", 1)
        else:
            module_path, obj_name = path.rsplit(".", 1)

        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, obj_name)

    def __getitem__(self, name: str):  # noqa: ANN204
        """Allow dict-like access."""
        return self.get(name)

    def is_loaded(self, name: str) -> bool:
        """Check if a module has been loaded."""
        return name in self._loaded

    def unload(self, name: str):
        """Unload a module from cache."""
        if name in self._loaded:
            del self._loaded[name]

    def keys(self) -> list[str]:
        """List all registered modules."""
        return list(self._registry.keys())


if __name__ == "__main__":
    from collections import defaultdict

    # usage example
    registry = ModuleRegistry()

    # register modules
    registry.register("json", "json")
    registry.register("datetime", "datetime:datetime")
    registry.register("pathlib", "pathlib:Path")

    # custom factory function
    def create_custom_handler() -> defaultdict:
        return defaultdict(list)
        # registry.register("handler", None, factory=create_custom_handler)

    # modules are only imported when accessed
    json_module = registry["json"]  # imports json now
    dt = registry["datetime"]  # imports datetime now
