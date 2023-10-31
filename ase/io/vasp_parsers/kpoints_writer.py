from dataclasses import dataclass
from typing import Union, List


@dataclass
class _State:
    number_kpoints: int = 0
    mode: Union[str, None] = None
    coordinate: Union[str, None] = None
    specification: Union[List[str], None] = None


def write_kpoints(directory, parameters):
    if isinstance(parameters, str):
        kpoints_str = parameters
    elif parameters is None:
        return
    else:
        state = _State()
        for item in parameters.items():
            state = update_state(state, *item)
        kpoints_str = "\n".join(prepare_lines(state))
    with open(f"{directory}/KPOINTS", "w") as kpoints:
        kpoints.write(kpoints_str)


def update_state(state, key, value):
    key = key.capitalize()
    if key == "Auto":
        state.mode = key
        state.specification = [str(value)]
    elif key in ("Gamma", "Monkhorst"):
        state.mode = key
        state.specification = [" ".join(str(x) for x in value)]
    elif key == "Line":
        state.mode = key
        state.number_kpoints = value
    elif key in ("Reciprocal", "Cartesian"):
        if state.number_kpoints == 0:
            state.number_kpoints = len(value)
        state.coordinate = key
        state.specification = [" ".join(str(x) for x in kpt) for kpt in value]
    else:
        raise NotImplementedError
    return state


def prepare_lines(state):
    yield "KPOINTS created by Atomic Simulation Environment"
    yield str(state.number_kpoints)
    if state.mode is not None:
        yield state.mode
    if state.coordinate is not None:
        yield state.coordinate
    for line in state.specification:
        yield line
