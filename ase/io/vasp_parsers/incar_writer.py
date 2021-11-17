from collections.abc import Iterable

def write_incar(parameters):
    incar_lines = []
    for key, value in parameters.items():
        if isinstance(value,str):
            if value.find("\n") != -1:
                value = '"' + value + '"'
            incar_lines.append(f"{key} = {value}")
        elif isinstance(value,Iterable):
            incar_lines.append(f"{key} = {' '.join(str(x) for x in value)}")
        else:
            incar_lines.append(f"{key} = {value}")

    with open("INCAR", "w") as incar:
        incar.write("\n".join(incar_lines))
