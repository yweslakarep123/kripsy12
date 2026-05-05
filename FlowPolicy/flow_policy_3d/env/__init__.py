__all__ = ["AdroitEnv", "MetaWorldEnv", "FrankaKitchenPointCloudEnv"]


def __getattr__(name):
    if name == "AdroitEnv":
        from .adroit import AdroitEnv

        return AdroitEnv
    if name == "MetaWorldEnv":
        from .metaworld import MetaWorldEnv

        return MetaWorldEnv
    if name == "FrankaKitchenPointCloudEnv":
        from .franka_kitchen import FrankaKitchenPointCloudEnv

        return FrankaKitchenPointCloudEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
