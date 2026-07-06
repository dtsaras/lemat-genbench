import json
import math
import re
from collections import defaultdict
from itertools import combinations_with_replacement, product
from pathlib import Path

import numpy as np
from pymatgen.analysis.bond_valence import BVAnalyzer, calculate_bv_sum
from pymatgen.analysis.local_env import get_neighbors_of_site_with_index
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element, Species
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from lemat_genbench.utils.logging import logger


OXIDATION_STATE_RECORD_KEY = "oxidation_state_record"
MIGRATION_BARRIER_READY_KEY = "migration_barrier_ready"


def electronegativity_correlation(
        elements: list[str],
        oxidation_states: list[int | float]
        ) -> float:
    """
    Calculate correlation between oxidation states and electronegativity.
    
    Args:
        elements: List of element symbols
        oxidation_states: List of oxidation state values (averaged per element)
        TODO this should probably be scaled to follow the number of elements with that 
        oxidation state.
        
    Returns:
        Pearson correlation coefficient between oxidation states and electronegativity.
        Returns NaN if correlation cannot be calculated.
    """

    en_vals = []
    for el in elements: 
        try:
            en_vals.append(Element(el).X)
        except (KeyError, AttributeError):
            # Use Pauling scale default if not available, or raise error
            logger.warning(f"No electronegativity data for element {el}")
            return np.nan  # Can't calculate correlation without complete data

    if len(en_vals) != len(oxidation_states):
        logger.error("Mismatch in array lengths for correlation calculation")
        return np.nan
    else:
        corr = np.corrcoef(oxidation_states, en_vals)[0,1]
        return corr


def compositional_oxi_state_guesses(
    comp,
    all_oxi_states: bool,
    max_sites: int | None,
    oxi_states_override: dict[str, list] | None,
    target_charge: float,
) -> tuple[tuple, tuple, tuple]:
    """Utility operation for guessing oxidation states. 
    Adapted from the _get_oxi_state_guesses function from Pymatgen.core.Composition

    See `oxi_state_guesses` for full details. This operation does the
    calculation of the most likely oxidation states

    Args:
        comp: A Pymatgen composition object.
        oxi_states_override (dict): dict of str->list to override an element's common oxidation states, e.g.
            {"V": [2,3,4,5]}.
        target_charge (float): the desired total charge on the structure. Default is 0 signifying charge balance.
        all_oxi_states (bool): if True, all oxidation states of an element, even rare ones, are used in the search
            for guesses. However, the full oxidation state list is *very* inclusive and can produce nonsensical
            results. If False, the icsd_oxidation_states list is used when present, or the common_oxidation_states
            is used when icsd_oxidation_states is not present. These oxidation states lists comprise more
            commonly occurring oxidation states and results in more reliable guesses, albeit at the cost of
            missing some uncommon situations. The default is False.
        max_sites (int): if possible, will reduce Compositions to at most
            this many sites to speed up oxidation state guesses. If the
            composition cannot be reduced to this many sites a ValueError
            will be raised. Set to -1 to just reduce fully. If set to a
            number less than -1, the formula will be fully reduced but a
            ValueError will be thrown if the number of atoms in the reduced
            formula is greater than abs(max_sites).

    Returns:
        list[dict]: Each dict maps the element symbol to a list of
            oxidation states for each site of that element. For example, Fe3O4 could
            return a list of [2,2,2,3,3,3] for the oxidation states of the 6 Fe sites.
            If the composition is not charge balanced, an empty list is returned.
    """
    # Reduce Composition if necessary
    if max_sites and max_sites < 0:
        comp = comp.reduced_composition

        if max_sites < -1 and comp.num_atoms > abs(max_sites):
            raise ValueError(
                f"Composition {comp} cannot accommodate max_sites setting!"
            )

    elif max_sites and comp.num_atoms > max_sites:
        reduced_comp, reduced_factor = comp.get_reduced_composition_and_factor()
        if reduced_factor > 1:
            reduced_comp *= max(1, int(max_sites / reduced_comp.num_atoms))
            comp = reduced_comp  # as close to max_sites as possible
        if comp.num_atoms > max_sites:
            raise ValueError(
                f"Composition {comp} cannot accommodate max_sites setting!"
            )

    # Load prior probabilities of oxidation states, used to rank solutions
    here = Path(__file__).resolve().parent
    three_up = here.parents[2]

    # Try loading from repo root first, fallback to package data directory
    oxi_probs_path = three_up / "data" / "lemat_icsd_oxi_dict_probs.json"
    if not oxi_probs_path.exists():
        # Fallback: load from package data directory (works when installed as package)
        package_data_dir = here.parent / "data"
        oxi_probs_path = package_data_dir / "lemat_icsd_oxi_dict_probs.json"

    with open(oxi_probs_path, "r") as f:
        loaded_dict = json.load(f)
    type(comp).oxi_prob = loaded_dict
    oxi_states_override = oxi_states_override or {}
    # Assert Composition only has integer amounts
    if not all(amt == int(amt) for amt in comp.values()):
        raise ValueError(
            "Charge balance analysis requires integer values in Composition!"
        )

    # For each element, determine all possible sum of oxidations
    # (taking into account nsites for that particular element)
    el_amt = comp.get_el_amt_dict()
    n_sites = int(sum(el_amt.values()))
    elements = list(el_amt)
    el_sums: list = []  # matrix: dim1= el_idx, dim2=possible sums
    el_sum_scores: defaultdict = defaultdict(set)  # dict of el_idx, sum -> score
    el_best_oxid_combo: dict = {}  # dict of el_idx, sum -> oxid combo with best score

    for idx, el in enumerate(elements):
        el_sum_scores[idx] = {}
        el_best_oxid_combo[idx] = {}
        el_sums.append([])
        if oxi_states_override.get(el):
            oxids: list | tuple = oxi_states_override[el]
        elif all_oxi_states:
            oxids = Element(el).oxidation_states
        else:
            oxids = (
                Element(el).icsd_oxidation_states or Element(el).common_oxidation_states
            )

        # Get all possible combinations of oxidation states
        # and sum each combination
        for oxid_combo in combinations_with_replacement(oxids, int(el_amt[el])):
            # check to make sure none of the oxidation states deviate by more than 1 
            if max(oxid_combo) - min(oxid_combo) <= 1: 
                # List this sum as a possible option
                oxid_sum = sum(oxid_combo)
                if oxid_sum not in el_sums[idx]:
                    el_sums[idx].append(oxid_sum)

                # Determine how probable is this combo?
                if not all_oxi_states:
                    scores = []
                    for o in oxid_combo:
                        scores.append(type(comp).oxi_prob[str(Species(el, o))])
                    score = math.prod(scores)

                    # If it is the most probable combo for a certain sum,
                    # store the combination
                    if oxid_sum not in el_sum_scores[idx] or score > el_sum_scores[idx].get(
                        oxid_sum, 0):
                        
                        el_sum_scores[idx][oxid_sum] = score
                        el_best_oxid_combo[idx][oxid_sum] = oxid_combo
                            
            else:
                pass
    
    # Determine which combination of oxidation states for each element
    # is the most probable

    el_sums = [[x for x in sublist if x != 0] for sublist in el_sums]
    
    all_sols = []  # will contain all solutions
    all_oxid_combo = []  # will contain the best combination of oxidation states for each site
    all_scores = []  # will contain a score for each solution
    scores = []
    for x in product(*el_sums):
        # Each x is a trial of one possible oxidation sum for each element
        if sum(x) == target_charge:  # charge balance condition
            el_sum_sol = dict(zip(elements, x, strict=True))  # element->oxid_sum
            # Normalize oxid_sum by amount to get avg oxid state
            sol = {el: v / el_amt[el] for el, v in el_sum_sol.items()}
            # Add the solution to the list of solutions

                        
            all_sols.append(sol)
            
            if not all_oxi_states:
                # Determine the score for this solution
                scores = []
                for idx, v in enumerate(x):
                    scores.append(el_sum_scores[idx][v])
                # the score is geometric mean of the scores for all the elements in the compostion 
                all_scores.append(math.prod(scores)**(1/n_sites))
                # Collect the combination of oxidation states for each site
                all_oxid_combo.append(
                    {
                        e: el_best_oxid_combo[idx][v]
                        for idx, (e, v) in enumerate(zip(elements, x, strict=True))
                    }
                )
            else:
                all_scores.append(electronegativity_correlation(elements=list(sol.keys()), oxidation_states=list(sol.values())))


    # Sort the solutions from highest to lowest score
    if all_scores:
        if all_oxi_states:
            # For correlation: more negative is better (ascending sort)
            sorted_data = sorted(
                zip(all_scores, all_sols),
                key=lambda x: x[0]  # Sort by score
            )
            all_scores, all_sols = zip(*sorted_data)
            all_oxid_combo = all_sols
            return (
                tuple(all_sols),
                tuple(all_oxid_combo),
                tuple(all_scores),
            )

        else:
            # For probabilities: higher is better (descending sort)
            sorted_data = sorted(
                zip(all_scores, all_sols, all_oxid_combo),
                key=lambda x: x[0],
                reverse=True
            )
            all_scores, all_sols, all_oxid_combo = zip(*sorted_data)


            return (
                tuple(all_sols),
                tuple(all_oxid_combo),
                tuple(all_scores),
            )
    else:
        return (
            tuple(all_sols),
            tuple(all_oxid_combo),
            tuple(all_scores),
            )


def get_inequivalent_site_info(structure):
    """Gets the symmetrically inequivalent sites as found by the
    SpacegroupAnalyzer class from Pymatgen.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        The Pymatgen structure of interest.

    Returns
    -------
    dict
        A dictionary containing three lists, one of the inequivalent sites, one
        for the atom types they correspond to and the last for the multiplicity.
    """

    # Get the symmetrically inequivalent indexes
    inequivalent_sites = (
        SpacegroupAnalyzer(structure).get_symmetrized_structure().equivalent_indices
    )

    # Equivalent indexes must all share the same atom type
    multiplicities = [len(xx) for xx in inequivalent_sites]
    inequivalent_sites = [xx[0] for xx in inequivalent_sites]
    species = [str(structure[xx].specie) for xx in inequivalent_sites]

    return {
        "sites": inequivalent_sites,
        "species": species,
        "multiplicities": multiplicities,
    }


def build_oxi_dict(df):
    oxi_dict = {}
    for i in range(0, len(df)):
        row = df.iloc[i]
        if row.ValencesCalculated:
            for key, value in np.asarray(
                [row.Sites["species"], row.Sites["multiplicities"]]
            ).T:
                if key in oxi_dict:
                    oxi_dict[key] += int(value)
                else:
                    oxi_dict[key] = int(value)
    return oxi_dict


def build_sorted_oxi_dict(oxi_dict_sorted):
    oxi_dict_counts = {}
    for key in oxi_dict_sorted.keys():
        try:
            int(key[1])
            el = key[0]
        except ValueError:
            if key[1] in ["+", "-"]:
                el = key[0]
            else:
                el = key[0:2]

        if el in oxi_dict_counts:
            oxi_dict_counts[el] += int(oxi_dict_sorted[key])
        else:
            oxi_dict_counts[el] = int(oxi_dict_sorted[key])
    return oxi_dict_counts


def build_oxi_dict_probs(oxi_dict_sorted, oxi_dict_counts):
    oxi_dict_probs = oxi_dict_sorted
    for key in oxi_dict_probs.keys():
        try:
            int(key[1])
            el = key[0]
        except ValueError:
            if key[1] in ["+", "-"]:
                el = key[0]
            else:
                el = key[0:2]
        denom = oxi_dict_counts[el]
        oxi_dict_probs[key] = oxi_dict_probs[key] / denom
    return oxi_dict_probs


def oxi_state_map(oxidation_state):
    try:
        int(oxidation_state[1])
        el = oxidation_state[0]
        ox = int(oxidation_state[1:3][::-1])
        return ox, el
    except ValueError:
        if oxidation_state[1] in ["+", "-"]:
            ox = sign_to_int(oxidation_state[1])
            el = oxidation_state[0]
            return ox, el
        elif oxidation_state[2] in ["+", "-"]:
            ox = sign_to_int(oxidation_state[2])
            el = oxidation_state[0:2]
            return ox, el
        else:
            ox = int(oxidation_state[2:4][::-1])
            el = oxidation_state[0:2]
            return ox, el


def build_oxi_state_map(oxi_dict_sorted):
    oxi_state_mapping = {}
    for key in oxi_dict_sorted.keys():
        ox, el = oxi_state_map(key)
        if el in oxi_state_mapping:
            oxi_state_mapping[el].append(ox)
        else:
            oxi_state_mapping[el] = [ox]
    return oxi_state_mapping


def sign_to_int(char):
    return {"+": 1, "-": -1}.get(char, 0)  # default to 0 if unexpected


def parse_mobile_ion(mobile_ion: str) -> tuple[str, float | None]:
    """Parse a pymatgen-style mobile ion string, e.g. ``Li1+`` -> (Li, 1)."""
    match = re.fullmatch(r"([A-Z][a-z]?)(?:(\d+(?:\.\d+)?)([+-])|([+-]))?", mobile_ion)
    if not match:
        return mobile_ion, None
    symbol, mag, sign, bare_sign = match.groups()
    if sign or bare_sign:
        charge = float(mag or 1)
        if (sign or bare_sign) == "-":
            charge *= -1
        return symbol, charge
    return symbol, None


def _site_symbol(site) -> str:
    specie = site.specie
    return str(getattr(specie, "symbol", specie))


def _site_oxi_state(site) -> float | None:
    oxi = getattr(site.specie, "oxi_state", None)
    return None if oxi is None else float(oxi)


def _round_num(value: float | None) -> float | None:
    if value is None or np.isnan(value):
        return None
    rounded = round(float(value), 6)
    return int(rounded) if rounded == int(rounded) else rounded


def _site_records_from_structure(structure: Structure) -> list[dict]:
    records = []
    for idx, site in enumerate(structure.sites):
        oxi = _site_oxi_state(site)
        if oxi is None:
            return []
        records.append(
            {
                "site_index": idx,
                "element": _site_symbol(site),
                "oxi_state": _round_num(oxi),
            }
        )
    return records


def _states_by_element(site_records: list[dict]) -> dict:
    grouped: dict[str, list] = defaultdict(list)
    for rec in site_records:
        grouped[rec["element"]].append(rec["oxi_state"])
    out = {}
    for element, values in grouped.items():
        unique = sorted(set(values))
        out[element] = unique[0] if len(unique) == 1 else unique
    return out


def decorated_structure_from_oxidation_record(
    structure: Structure, record: dict | None = None
) -> Structure:
    """Return a copy decorated with oxidation states stored in a record."""
    record = record or structure.properties.get(OXIDATION_STATE_RECORD_KEY)
    if not record or not record.get("oxidation_states_by_site"):
        raise ValueError("No oxidation-state record available")

    site_states = record["oxidation_states_by_site"]
    if len(site_states) != len(structure):
        raise ValueError("Oxidation-state record does not match structure length")

    decorated = structure.copy()
    try:
        decorated.remove_oxidation_states()
    except Exception:
        pass
    decorated.add_oxidation_state_by_site(
        [float(rec["oxi_state"]) for rec in site_states]
    )
    return decorated


def _candidate_from_structure(source: str, decorated: Structure, score=None) -> dict:
    site_records = _site_records_from_structure(decorated)
    if not site_records:
        raise ValueError("candidate has no site-specific oxidation states")
    return {
        "source": source,
        "oxidation_states_by_site": site_records,
        "oxidation_states_by_element": _states_by_element(site_records),
        "prior_score": _round_num(score) if score is not None else None,
    }


def _candidate_from_combo(structure: Structure, source: str, combo: dict, score=None) -> dict:
    counts: defaultdict[str, int] = defaultdict(int)
    site_states = []
    for idx, site in enumerate(structure.sites):
        element = _site_symbol(site)
        values = combo.get(element)
        if values is None:
            raise ValueError(f"missing oxidation state for {element}")
        if isinstance(values, (list, tuple)):
            pos = min(counts[element], len(values) - 1)
            oxi_state = values[pos]
            counts[element] += 1
        else:
            oxi_state = values
        site_states.append(
            {
                "site_index": idx,
                "element": element,
                "oxi_state": _round_num(float(oxi_state)),
            }
        )
    return {
        "source": source,
        "oxidation_states_by_site": site_states,
        "oxidation_states_by_element": _states_by_element(site_states),
        "prior_score": _round_num(score) if score is not None else None,
    }


def _load_oxi_state_mapping() -> dict:
    here = Path(__file__).resolve().parent
    three_up = here.parents[2]
    path = three_up / "data" / "lemat_icsd_oxi_state_mapping.json"
    if not path.exists():
        path = here.parent / "data" / "lemat_icsd_oxi_state_mapping.json"
    with open(path, "r") as f:
        return json.load(f)


def _composition_candidates(
    structure: Structure, max_candidates: int
) -> list[dict]:
    comp = Composition(structure.composition.element_composition)
    oxi_state_mapping = _load_oxi_state_mapping()
    overrides = {
        str(el): oxi_state_mapping[str(el)]
        for el in comp.elements
        if str(el) in oxi_state_mapping
    }
    candidates = []

    for all_oxi_states, source in (
        (False, "composition_guess"),
        (True, "composition_guess_all_states"),
    ):
        try:
            _, combos, scores = compositional_oxi_state_guesses(
                comp,
                all_oxi_states=all_oxi_states,
                max_sites=-1,
                target_charge=0,
                oxi_states_override=None if all_oxi_states else overrides,
            )
        except Exception as exc:
            logger.debug("Composition oxidation-state guesses failed: %s", exc)
            continue

        for combo, score in list(zip(combos, scores))[:max_candidates]:
            try:
                candidates.append(_candidate_from_combo(structure, source, combo, score))
            except Exception as exc:
                logger.debug("Could not build oxidation candidate: %s", exc)
        if candidates:
            break
    return candidates


def _compute_bvs_mismatch(structure: Structure, candidate: dict) -> dict:
    decorated = decorated_structure_from_oxidation_record(structure, candidate)
    mismatches = []
    mobile = []
    for rec in candidate["oxidation_states_by_site"]:
        idx = rec["site_index"]
        try:
            nn_list = get_neighbors_of_site_with_index(decorated, idx)
            bvs = calculate_bv_sum(decorated[idx], nn_list)
            mismatch = abs(abs(float(bvs)) - abs(float(rec["oxi_state"])))
            mismatches.append(mismatch)
            if rec.get("_is_mobile_ion"):
                mobile.append(mismatch)
        except Exception as exc:
            logger.debug("BVS mismatch failed for site %s: %s", idx, exc)

    if not mismatches:
        return {
            "mean_abs_mismatch": None,
            "max_abs_mismatch": None,
            "p90_abs_mismatch": None,
            "mobile_ion_mean_abs_mismatch": None,
        }
    return {
        "mean_abs_mismatch": _round_num(float(np.mean(mismatches))),
        "max_abs_mismatch": _round_num(float(np.max(mismatches))),
        "p90_abs_mismatch": _round_num(float(np.percentile(mismatches, 90))),
        "mobile_ion_mean_abs_mismatch": _round_num(float(np.mean(mobile)))
        if mobile
        else None,
    }


def _chemical_failures(site_records: list[dict]) -> list[str]:
    failures = []
    alkali = {"Li", "Na", "K", "Rb", "Cs"}
    alkaline = {"Mg", "Ca", "Sr", "Ba"}
    for rec in site_records:
        el = rec["element"]
        ox = float(rec["oxi_state"])
        if el in alkali and ox != 1:
            failures.append(f"{el} expected +1, got {ox:g}")
        elif el in alkaline and ox != 2:
            failures.append(f"{el} expected +2, got {ox:g}")
        elif el == "F" and ox != -1:
            failures.append(f"F expected -1, got {ox:g}")
        elif el == "O" and ox > 0:
            failures.append(f"O has positive oxidation state {ox:g}")
    return failures


def _check_bvlain_candidate(
    structure: Structure,
    candidate: dict,
    mobile_ion: str,
    check_bvlain_parameters: bool | str,
    bvlain_settings: dict | None,
) -> tuple[bool | None, list[str], list[str]]:
    if not check_bvlain_parameters:
        return None, ["bvlain parameter check skipped"], []

    try:
        from bvlain import Lain
    except ImportError:
        msg = "bvlain is not installed"
        if check_bvlain_parameters == "auto":
            return None, [msg], []
        return False, [], [msg]

    bvlain_settings = bvlain_settings or {}
    try:
        decorated = decorated_structure_from_oxidation_record(structure, candidate)
        calc = Lain(verbose=False)
        calc.read_structure(decorated, oxi_check=False)
        calc.bvse_distribution(
            mobile_ion=mobile_ion,
            r_cut=bvlain_settings.get("r_cut", 10.0),
            resolution=bvlain_settings.get("resolution", 0.2),
            k=bvlain_settings.get("k", 100),
        )
        return True, [], []
    except Exception as exc:
        return False, [], [f"bvlain parameter check failed: {exc}"]


def _validate_oxidation_candidate(
    structure: Structure,
    candidate: dict,
    *,
    mobile_ion: str,
    charge_tolerance: float,
    require_mobile_ion: bool,
    check_bvlain_parameters: bool | str,
    bvlain_settings: dict | None,
) -> dict:
    site_records = candidate["oxidation_states_by_site"]
    mobile_symbol, mobile_charge = parse_mobile_ion(mobile_ion)
    failures = []
    warnings = []

    total_charge = sum(float(rec["oxi_state"]) for rec in site_records)
    if abs(total_charge) > charge_tolerance:
        failures.append(f"total charge not neutral: {total_charge:g}")

    mobile_records = [rec for rec in site_records if rec["element"] == mobile_symbol]
    for rec in mobile_records:
        rec["_is_mobile_ion"] = True
    if require_mobile_ion and not mobile_records:
        failures.append(f"mobile ion {mobile_symbol} not present")
    if mobile_charge is not None:
        for rec in mobile_records:
            if abs(float(rec["oxi_state"]) - mobile_charge) > charge_tolerance:
                failures.append(
                    f"mobile ion {mobile_ion} mismatch at site "
                    f"{rec['site_index']}: {rec['oxi_state']}"
                )

    failures.extend(_chemical_failures(site_records))
    mismatch = _compute_bvs_mismatch(structure, candidate)
    bvlain_complete, bvlain_warnings, bvlain_failures = _check_bvlain_candidate(
        structure,
        candidate,
        mobile_ion=mobile_ion,
        check_bvlain_parameters=check_bvlain_parameters,
        bvlain_settings=bvlain_settings,
    )
    warnings.extend(bvlain_warnings)
    failures.extend(bvlain_failures)

    return {
        **candidate,
        **mismatch,
        "total_charge": _round_num(total_charge),
        "passed": not failures,
        "bvlain_parameters_complete": bvlain_complete,
        "warnings": warnings,
        "failure_reasons": failures,
    }


def _confidence_label(candidate: dict) -> str:
    if not candidate.get("passed"):
        return "failed"
    mean = candidate.get("mean_abs_mismatch")
    max_m = candidate.get("max_abs_mismatch")
    warnings = candidate.get("warnings") or []
    if mean is None or max_m is None:
        return "usable_with_warning"
    if mean <= 0.20 and max_m <= 0.60 and not warnings:
        return "high_confidence"
    if mean <= 0.40 and max_m <= 1.00:
        return "medium_confidence"
    if mean <= 0.60 and max_m <= 1.50:
        return "low_confidence"
    return "usable_with_warning"


def assign_oxidation_states_for_bvlain(
    structure: Structure,
    *,
    mobile_ion: str = "Li1+",
    charge_tolerance: float = 1e-3,
    require_mobile_ion: bool = False,
    min_num_elements: int | None = None,
    check_bvlain_parameters: bool | str = False,
    bvlain_settings: dict | None = None,
    max_candidates: int = 16,
) -> dict:
    """Assign oxidation states and return a JSON-serializable readiness record."""
    elements = sorted(
        str(getattr(el, "symbol", el)) for el in structure.composition.elements
    )
    mobile_symbol, _ = parse_mobile_ion(mobile_ion)
    precheck_failures = []
    if require_mobile_ion and mobile_symbol not in elements:
        precheck_failures.append(f"mobile ion {mobile_symbol} not present")
    if min_num_elements is not None and len(elements) < min_num_elements:
        precheck_failures.append(
            f"number of elements {len(elements)} is less than required "
            f"minimum {min_num_elements}"
        )
    if precheck_failures:
        return {
            "status": "failed",
            "selected_source": None,
            "confidence_label": "failed",
            "mobile_ion": mobile_ion,
            "elements": elements,
            "num_elements": len(elements),
            "min_num_elements": min_num_elements,
            "migration_barrier_ready": False,
            "failure_reasons": precheck_failures,
            "warnings": [],
            "candidate_summary": [],
        }

    candidates = []

    if _site_records_from_structure(structure):
        try:
            candidates.append(_candidate_from_structure("cif", structure))
        except Exception as exc:
            logger.debug("Could not use existing oxidation states: %s", exc)

    try:
        candidates.append(
            _candidate_from_structure(
                "bvanalyzer", BVAnalyzer().get_oxi_state_decorated_structure(structure)
            )
        )
    except Exception as exc:
        logger.debug("BVAnalyzer oxidation-state assignment failed: %s", exc)

    try:
        candidates.extend(_composition_candidates(structure, max_candidates))
    except Exception as exc:
        logger.debug("Composition oxidation-state candidates failed: %s", exc)

    seen = set()
    unique_candidates = []
    for cand in candidates:
        key = tuple(rec["oxi_state"] for rec in cand["oxidation_states_by_site"])
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)

    evaluated = [
        _validate_oxidation_candidate(
            structure,
            cand,
            mobile_ion=mobile_ion,
            charge_tolerance=charge_tolerance,
            require_mobile_ion=require_mobile_ion,
            check_bvlain_parameters=check_bvlain_parameters,
            bvlain_settings=bvlain_settings,
        )
        for cand in unique_candidates[:max_candidates]
    ]

    source_priority = {
        "cif": 0,
        "bvanalyzer": 1,
        "composition_guess": 2,
        "composition_guess_all_states": 3,
    }

    def rank_key(cand: dict):
        return (
            cand.get("mean_abs_mismatch")
            if cand.get("mean_abs_mismatch") is not None
            else float("inf"),
            cand.get("max_abs_mismatch")
            if cand.get("max_abs_mismatch") is not None
            else float("inf"),
            cand.get("mobile_ion_mean_abs_mismatch")
            if cand.get("mobile_ion_mean_abs_mismatch") is not None
            else float("inf"),
            source_priority.get(cand["source"], 99),
        )

    valid_candidates = [cand for cand in evaluated if cand["passed"]]
    selected = sorted(valid_candidates, key=rank_key)[0] if valid_candidates else None
    if selected is None:
        reasons = []
        for cand in evaluated:
            reasons.extend(cand.get("failure_reasons", []))
        return {
            "status": "failed",
            "selected_source": None,
            "confidence_label": "failed",
            "mobile_ion": mobile_ion,
            "elements": elements,
            "num_elements": len(elements),
            "min_num_elements": min_num_elements,
            "migration_barrier_ready": False,
            "failure_reasons": sorted(set(reasons)) or ["no oxidation-state candidate"],
            "warnings": [],
            "candidate_summary": _candidate_summary(evaluated),
        }

    bvlain_complete = selected.get("bvlain_parameters_complete")
    ready = selected["passed"] and bvlain_complete is not False
    return {
        "status": "success",
        "selected_source": selected["source"],
        "confidence_label": _confidence_label(selected),
        "mobile_ion": mobile_ion,
        "elements": elements,
        "num_elements": len(elements),
        "min_num_elements": min_num_elements,
        "total_charge": selected["total_charge"],
        "mean_abs_mismatch": selected["mean_abs_mismatch"],
        "max_abs_mismatch": selected["max_abs_mismatch"],
        "p90_abs_mismatch": selected["p90_abs_mismatch"],
        "mobile_ion_mean_abs_mismatch": selected["mobile_ion_mean_abs_mismatch"],
        "oxidation_states_by_element": selected["oxidation_states_by_element"],
        "oxidation_states_by_site": [
            {k: v for k, v in rec.items() if not k.startswith("_")}
            for rec in selected["oxidation_states_by_site"]
        ],
        "bvlain_parameters_complete": bvlain_complete,
        "missing_bvlain_parameters": selected["failure_reasons"]
        if bvlain_complete is False
        else [],
        "migration_barrier_ready": ready,
        "warnings": selected["warnings"],
        "failure_reasons": selected["failure_reasons"],
        "candidate_summary": _candidate_summary(evaluated),
    }


def _candidate_summary(candidates: list[dict]) -> list[dict]:
    return [
        {
            "source": cand.get("source"),
            "passed": cand.get("passed", False),
            "mean_abs_mismatch": cand.get("mean_abs_mismatch"),
            "max_abs_mismatch": cand.get("max_abs_mismatch"),
            "bvlain_parameters_complete": cand.get("bvlain_parameters_complete"),
            "warnings": cand.get("warnings", []),
            "failure_reasons": cand.get("failure_reasons", []),
        }
        for cand in candidates
    ]
