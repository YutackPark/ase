def write_incar(parameters):
    with open("INCAR", "w") as incar:
        for key, value in parameters.items():
            incar.write(f"{key} = {value}")
