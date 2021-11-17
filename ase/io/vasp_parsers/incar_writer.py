from collections.abc import Iterable

def write_incar(parameters):
    incar_lines = []
    for key, value in parameters.items():
        if isinstance(value,Iterable) and not isinstance(value,str):
            incar_lines.append(f"{key} = {' '.join(str(x) for x in value)}")
        else:
            incar_lines.append(f"{key} = {value}")

    with open("INCAR", "w") as incar:
        incar.write("\n".join(incar_lines))
