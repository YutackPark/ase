from ase.calculators.espresso import EspressoTemplate
from ase.calculators.abinit import AbinitTemplate

from ase.config import Config


def test_socketio_mpi_generator():
    cfg = Config()
    cfg.parser["parallel"] = {"binary": "mpirun"}
    cfg.parser["espresso"] = {"binary": "pw.x", "pseudo_path": "test"}
    cfg.parser["abinit"] = {"binary": "abinit"}

    for temp_class in [EspressoTemplate, AbinitTemplate]:
        template = temp_class()
        profile = template.load_profile(cfg)

        socket_argv = template.socketio_argv(profile, "UNIX:TEST", None)
        profile_command = profile.get_command(
            inputfile=None,
            calc_command=socket_argv
        )
        print(profile_command)
        assert all(
            [test == ref
             for test, ref in zip(profile_command, ["mpirun"] + socket_argv)]
        )
