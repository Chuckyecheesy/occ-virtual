"""Manages Snowflake tag."""

from ._generated.models import (
    Tag,
)
from ._tag import TagCollection, TagResource


__all__ = [
    "TagResource",
    "TagCollection",
    "Tag",
]
