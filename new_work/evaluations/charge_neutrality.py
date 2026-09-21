"""Utilities for selecting charge-neutral structures."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser


OXIDATION_STATES_DICT = {
    "H": [-1, 1],
    "Li": [-1, 1],
    "Be": [1, 2],
    "B": [-5, -1, 1, 2, 3],
    "C": [-4, -3, -2, -1, 1, 2, 3, 4],
    "N": [-3, -2, -1, 1, 2, 3, 4, 5],
    "O": [-2, -1, 1, 2],
    "F": [-1],
    "Na": [-1, 1],
    "Mg": [1, 2],
    "Al": [-2, -1, 1, 2, 3],
    "Si": [-4, -3, -2, -1, 1, 2, 3, 4],
    "P": [-3, -2, -1, 1, 2, 3, 4, 5],
    "S": [-2, -1, 1, 2, 3, 4, 5, 6],
    "Cl": [-1, 1, 2, 3, 4, 5, 6, 7],
    "K": [-1, 1],
    "Ca": [1, 2],
    "Sc": [1, 2, 3],
    "Ti": [-2, -1, 1, 2, 3, 4],
    "V": [-3, -1, 1, 2, 3, 4, 5],
    "Cr": [-4, -2, -1, 1, 2, 3, 4, 5, 6],
    "Mn": [-3, -2, -1, 1, 2, 3, 4, 5, 6, 7],
    "Fe": [-4, -2, -1, 1, 2, 3, 4, 5, 6, 7],
    "Co": [-3, -1, 1, 2, 3, 4, 5],
    "Ni": [-2, -1, 1, 2, 3, 4],
    "Cu": [-2, 1, 2, 3, 4],
    "Zn": [-2, 1, 2],
    "Ga": [-5, -4, -3, -2, -1, 1, 2, 3],
    "Ge": [-4, -3, -2, -1, 1, 2, 3, 4],
    "As": [-3, -2, -1, 1, 2, 3, 4, 5],
    "Se": [-2, -1, 1, 2, 3, 4, 5, 6],
    "Br": [-1, 1, 2, 3, 4, 5, 7],
    "Rb": [-1, 1],
    "Sr": [1, 2],
    "Y": [1, 2, 3],
    "Zr": [-2, 1, 2, 3, 4],
    "Nb": [-3, -1, 1, 2, 3, 4, 5],
    "Mo": [-4, -2, -1, 1, 2, 3, 4, 5, 6],
    "Tc": [-1, 1, 2, 3, 4, 5, 6, 7],
    "Ru": [-2, 1, 2, 3, 4, 5, 6, 7, 8],
    "Rh": [-3, -1, 1, 2, 3, 4, 5, 6, 7],
    "Pd": [1, 2, 3, 4, 5],
    "Ag": [-2, -1, 1, 2, 3],
    "Cd": [-2, 1, 2],
    "In": [-5, -2, -1, 1, 2, 3],
    "Sn": [-4, -3, -3, -2, 1, 2, 3, 4],
    "Sb": [-3, -2, -1, 1, 2, 3, 4, 5],
    "Te": [-2, -1, 1, 2, 3, 4, 5, 6],
    "I": [-1, 1, 2, 3, 4, 5, 6, 7],
    "Cs": [-1, 1],
    "Ba": [1, 2],
    "Hf": [-2, 1, 2, 3, 4],
    "Ta": [-3, -1, 1, 2, 3, 4, 5],
    "W": [-4, -2, -1, 1, 2, 3, 4, 5, 6],
    "Re": [-3, -1, 1, 2, 3, 4, 5, 6, 7],
    "Os": [-4, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8],
    "Ir": [-3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Pt": [-3, -2, -1, 1, 2, 3, 4, 5, 6],
    "Au": [-3, -2, -1, 1, 2, 3, 5],
    "Hg": [-2, 1, 2],
    "Tl": [-5, -2, -1, 1, 2, 3],
    "Pb": [-4, -2, -1, 1, 2, 3, 4],
    "Bi": [-3, -2, -1, 1, 2, 3, 4, 5],
    "Po": [-2, 2, 4, 5, 6],
    "At": [-1, 1, 3, 5, 7],
}


def structure_from_cif_string(cif_string: str) -> Structure:
    return CifParser.from_str(cif_string).parse_structures(primitive=True)[0]


def _coerce_structure(structure_or_cif: Structure | str) -> Structure:
    if isinstance(structure_or_cif, Structure):
        return structure_or_cif
    return structure_from_cif_string(structure_or_cif)


def is_charge_neutral(
    structure_or_cif: Structure | str,
    oxidation_states_dict: dict[str, list[int]] = OXIDATION_STATES_DICT,
) -> bool:
    """Return True if any oxidation-state assignment makes the structure neutral."""

    struct = _coerce_structure(structure_or_cif)
    composition = struct.composition

    if composition.is_element:
        return True

    elements = list(composition.elements)
    amounts = [composition[el] for el in elements]

    oxidation_state_lists = []
    for element in elements:
        oxidation_states = oxidation_states_dict.get(element.symbol)
        if not oxidation_states:
            return False
        oxidation_state_lists.append(oxidation_states)

    for oxidation_state_combo in product(*oxidation_state_lists):
        total_charge = sum(
            oxidation_state * amount
            for oxidation_state, amount in zip(oxidation_state_combo, amounts)
        )
        if abs(total_charge) < 1e-8:
            return True

    return False


def add_charge_neutral_column(
    df: pd.DataFrame,
    structure_column: str = "structures",
    output_column: str = "is_charge_neutral",
) -> pd.DataFrame:
    result = df.copy()
    result[output_column] = result[structure_column].apply(is_charge_neutral)
    return result


def select_charge_neutral_structures(
    df: pd.DataFrame,
    structure_column: str = "structures",
) -> pd.DataFrame:
    df_with_flag = add_charge_neutral_column(df, structure_column=structure_column)
    return df_with_flag[df_with_flag["is_charge_neutral"]].copy()


def main() -> None:
    csv_path = Path("prerelaxed_structures.csv")
    df_all = pd.read_csv(csv_path, index_col=0)

    df_all = add_charge_neutral_column(df_all)

    print(f"Total structures: {len(df_all)}")
    print(f"Charge-neutral structures: {df_all['is_charge_neutral'].sum()}")

    df_neutral = df_all[df_all["is_charge_neutral"]].copy()
    df_neutral["structures"] = df_neutral["structures"].apply(
        structure_from_cif_string
    )
    df_neutral["structures"] = df_neutral["structures"].apply(
        lambda structure: structure.to(fmt="cif")
    )

    output_path = Path("charge_neutral_structures.csv")
    df_neutral.to_csv(output_path)
    print(f"Saved charge-neutral rows to: {output_path}")


if __name__ == "__main__":
    main()