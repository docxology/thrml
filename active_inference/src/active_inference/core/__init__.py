"""Core active inference components and data structures."""

from .free_energy import expected_free_energy, variational_free_energy
from .generative_model import GenerativeModel, HierarchicalGenerativeModel
from .precision import Message, MessageType, Precision, PrecisionWeighting

__all__ = [
    "GenerativeModel",
    "HierarchicalGenerativeModel",
    "variational_free_energy",
    "expected_free_energy",
    "Precision",
    "PrecisionWeighting",
    "Message",
    "MessageType",
]
