from collections.abc import Iterable


def write_incar(parameters):
    with open("INCAR", "w") as incar:
        for key, value in parameters.items():
            if isinstance(value, Iterable) and not isinstance(value, str):
                incar.write(f"{key} = {' '.join(str(x) for x in value)}")
            else:
                incar.write(f"{key} = {value}")
