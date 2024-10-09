class LazyLoadedObject:
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.obj = None
        self.args = args
        self.kwargs = kwargs

    def initialize(self):
        self.obj = self.cls(*self.args, **self.kwargs)

    def __call__(self, *args, **kwargs):
        if self.obj is None:
            self.obj = self.cls(*self.args, **self.kwargs)
        return self.obj(*args, **kwargs)


class RegistryBaseClass:
    def __init__(self):
        self.__dict__["registry"] = {}

    def register(self, name, *args, **kwargs):
        def decorator(obj):
            self.registry[name] = LazyLoadedObject(obj, *args, **kwargs)
            return obj

        return decorator

    def initialize(self):
        for obj in self.registry.values():
            if isinstance(obj, LazyLoadedObject):
                obj.initialize()

    def __getattr__(self, name: str):
        return self.registry[name]

    def __setattr__(self, name: str, value):
        self.registry[name] = value


registry = RegistryBaseClass()

# below are modules to be registered
from . import clip, reward_pipeline, yolo
