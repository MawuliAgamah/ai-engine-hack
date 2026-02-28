from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .data_loader import GREEN_BOOLEAN_KEYS, LoadedPatch, PatchAttributes


@dataclass(frozen=True)
class PatchCell:
    x: int
    y: int
    attrs: PatchAttributes

    @property
    def zone_type(self) -> str:
        return self.attrs["zone-type"]

    @property
    def occupied(self) -> bool:
        return bool(self.attrs["occupied?"])

    @property
    def traversable(self) -> bool:
        return bool(self.attrs["traversable"])


def patch_is_green(attrs: Mapping[str, object]) -> bool:
    return any(bool(attrs.get(key, False)) for key in GREEN_BOOLEAN_KEYS)


def build_patch_cells(loaded_patches: list[LoadedPatch]) -> list[PatchCell]:
    return [PatchCell(x=patch.pxcor, y=patch.pycor, attrs=patch.attrs) for patch in loaded_patches]


def build_patch_index(cells: list[PatchCell]) -> dict[tuple[int, int], PatchCell]:
    return {(cell.x, cell.y): cell for cell in cells}
