"""Alternate API variants used only by the intermediate experiment.

These classes inherit from the main APIs in ``func_source_code`` and add a
small amount of experimental behaviour (e.g., injected transient errors on
specific methods).  They are not used by the main benchmark — only by tasks
whose ``involved_classes`` list references the variant class name.
"""
