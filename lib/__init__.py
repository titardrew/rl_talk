from lib.network import NatureCNN, MLP
import warnings

NETWORK_REGISTRY = {}

def get_network(name):
    net_cls = NETWORK_REGISTRY.get(name)
    if net_cls is None:
        raise RuntimeError(f"Network '{name}' was not registered.")
    return net_cls

def register_network(name, network_cls):
    if name in NETWORK_REGISTRY:
        warnings.warn(f"Network '{name}' had already been registered.")

    NETWORK_REGISTRY[name] = network_cls


register_network("vision", NatureCNN)
register_network("mlp", MLP)


__all__ = [
    "get_network",
    "register_network"
]

