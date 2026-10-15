class cached_property:
    def __init__(self, func):
        self.func = func
        self.attr_name = "_cached_" + func.__name__

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if hasattr(instance, self.attr_name):
            return getattr(instance, self.attr_name)
        value = self.func(instance)
        setattr(instance, self.attr_name, value)
        return value

