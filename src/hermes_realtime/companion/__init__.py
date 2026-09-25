"""Hermes-side voice companion: the voice archive's storage core.

Every module here runs inside Hermes's own process, where Hermes's environment supplies its
imports. They depend on the standard library only and import Hermes lazily, so neither
this package nor the base test suite needs Hermes installed.
"""
