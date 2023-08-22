# creates:  nitrogen.txt, ethane.txt, gold.txt
import os
import sys
import tempfile
import subprocess


def run_script_and_get_output(script):
    """Returns the stdout of executing the code in pythonfile
    as a string."""
    script = os.path.join(os.getcwd(), script)
    with tempfile.TemporaryDirectory() as tempdir:
        return subprocess.check_output([sys.executable, script],
                                       cwd=tempdir)


# Only save the parts relevant to thermochemistry
output = run_script_and_get_output('nitrogen.py')
output = output[output.find(b'Enthalpy'):]
with open('nitrogen.txt', 'wb') as f:
    f.write(output)

output = run_script_and_get_output('ethane.py')
output = output[output.find(b'Internal'):]
with open('ethane.txt', 'wb') as f:
    f.write(output)

output = run_script_and_get_output('gold.py')
output = output[output.find(b'Internal'):]
with open('gold.txt', 'wb') as f:
    f.write(output)
