r"""|Cosmology| <-> MRT I/O, using |Cosmology.read| and |Cosmology.write|.

This module provides functions to write/read a |Cosmology| object to/from an MRT file.
The functions are registered with ``readwrite_registry`` under the format name "ascii.mrt".

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

import astropy.cosmology.units as cu
import astropy.units as u
from astropy.cosmology._src.core import Cosmology
from astropy.cosmology._src.io.connect import readwrite_registry
from astropy.io.typing import PathLike, ReadableFileLike, WriteableFileLike
from astropy.table import QTable
from astropy.table.serialize import represent_mixins_as_columns

from .table import from_table, to_table

if TYPE_CHECKING:
    from collections.abc import Mapping

    from astropy.cosmology._src.typing import _CosmoT
    from astropy.io.typing import PathLike, ReadableFileLike, WriteableFileLike
    from astropy.table import Table

    _TableT = TypeVar("_TableT", "Table")

_FORMAT_TABLE = {
    "H0": "$$H_{0}$$",
    "Om0": "$$\\Omega_{m,0}$$",
    "Ode0": "$$\\Omega_{\\Lambda,0}$$",
    "Tcmb0": "$$T_{0}$$",
    "Neff": "$$N_{eff}$$",
    "m_nu": "$$m_{nu}$$",
    "Ob0": "$$\\Omega_{b,0}$$",
    "w0": "$$w_{0}$$",
    "wa": "$$w_{a}$$",
    "wz": "$$w_{z}$$",
    "wp": "$$w_{p}$$",
    "zp": "$$z_{p}$$",
}


def read_mrt(
    file: PathLike | ReadableFileLike[Table],
    index: int | str | None = None,
    *,
    move_to_meta: bool = False,
    cosmology: str | type[_CosmoT] | None = None,
    rename: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> _CosmoT:
    r"""Read a `~astropy.cosmology.Cosmology` from an MRT file.

    Parameters
    ----------



    """
    format = kwargs.pop("format", "ascii.mrt")
    if format != "ascii.mrt":
        raise ValueError(f"format must be 'ascii.mrt',not {format}")

    table = QTable.read(file, format="ascii.mrt", **kwargs)

    m_nu_data = []
    i, more_m_nu = 0, True
    while more_m_nu:
        cn = f"m_nu[{i}]"  # column name
        if cn not in table.colnames:
            more_m_nu = False
            continue

        m_nu_data.append(table[cn][0])
        table.remove_column(cn)

        i += 1

    col = (
        [m_nu_data]
        if not isinstance(m_nu_data[0], u.Quantity)
        else [u.Quantity(m_nu_data)]
    )
    table.add_column(col, name="m_nu", index=-2)

    # Build the cosmology from table, using the private backend.
    return from_table(
        table, index=index, move_to_meta=move_to_meta, cosmology=cosmology
    )


def write_mrt(
    cosmology: Cosmology,
    file: PathLike | WriteableFileLike[_TableT],
    *,
    overwrite: bool = False,
    cls: type[_TableT] = QTable,
    latex_names: bool = False,
    **kwargs: Any,
):
    format = kwargs.pop("format", "ascii.mrt")
    if format != "ascii.mrt":
        raise ValueError(f"format must be 'ascii.mrt', not {format}")

    table = represent_mixins_as_columns(
        to_table(cosmology, cls=cls, cosmology_in_meta=False)
    )

    # CDS can't serialize redshift units, so remove them  # TODO: fix this
    for k, col in table.columns.items():
        if col.unit is cu.redshift:
            table[k] <<= u.dimensionless_unscaled

    # Replace the m_nu column with individual columns
    if "m_nu" in table.colnames:
        m_nu = table["m_nu"]

        index = table.colnames.index("m_nu")
        table.remove_column("m_nu")
        m_nu = np.atleast_1d(m_nu)  # Ensure m_nu is an array
        if m_nu.shape == (1,):
            # If m_nu is a scalar, add a single column
            table.add_column(m_nu[0], name="m_nu[0]", index=index)
        else:
            # If m_nu is an array, add multiple columns with names 'm_nu_[i]'
            cols, names = tuple(
                zip(*((m, f"m_nu[{i}]") for i, m in enumerate(m_nu[0])))
            )
            table.add_columns(cols, names=names, indexes=(index,) * len(m_nu[0]))

    table.write(file, overwrite=overwrite, format="ascii.mrt", **kwargs)


def mrt_identify(
    origin: object, filepath: object, *args: object, **kwargs: object
) -> bool:
    """Identify if an object uses the HTML Table format.

    Parameters
    ----------
    origin : object
        Not used.
    filepath : object
        From where to read the Cosmology.
    *args : object
        Not used.
    **kwargs : object
        Not used.

    Returns
    -------
    bool
        If the filepath is a string ending with '.mrt'.
    """
    return isinstance(filepath, str) and filepath.endswith(".mrt")


readwrite_registry.register_reader("ascii.mrt", Cosmology, read_mrt)
readwrite_registry.register_writer("ascii.mrt", Cosmology, write_mrt)
readwrite_registry.register_identifier("ascii.mrt", Cosmology, mrt_identify)
