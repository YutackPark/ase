from collections.abc import Iterable


def write_incar(parameters):
    incar_string = generate_incar_file(parameters)
    with open("INCAR", "w") as incar:
        incar.write(incar_string)


def generate_incar_file(parameters):
    if isinstance(parameters, str):
        return parameters
    else:
        incar_lines = []
        for item in parameters.items():
            incar_lines += list(generate_line(*item))
        return "\n".join(incar_lines)


def generate_line(key, value, num_spaces=0):
    indent = " " * num_spaces
    if isinstance(value, str):
        if value.find("\n") != -1:
            value = '"' + value + '"'
        yield indent + f"{key} = {value}"
    elif isinstance(value, dict):
        yield indent + f"{key} {{"
        for item in value.items():
            yield from generate_line(*item, num_spaces + 4)
        yield indent + "}"
    elif isinstance(value, Iterable):
        yield indent + f"{key} = {' '.join(str(x) for x in value)}"
    else:
        yield indent + f"{key} = {value}"
