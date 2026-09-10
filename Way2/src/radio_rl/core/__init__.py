"""Core contracts shared by every layer.

Nothing in ``core`` imports torch, the environment, or any plugin — it holds
only the frozen physical constants, the plain-data types that flow between
layers, the plugin registry, and the config composer.
"""
