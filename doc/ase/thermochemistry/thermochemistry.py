# creates:  nitrogen.txt, ethane.txt, gold.txt
import io
import shutil
import runpy
import sys


def output_to_string(pythonfile):
    """Returns the stdout of executing the code in pythonfile
    as a string."""
    buffer = io.StringIO()
    sys.stdout = buffer
    runpy.run_path(pythonfile)
    sys.stdout = sys.__stdout__
    return buffer.getvalue()


# Only save the parts relevant to thermochemistry
nitrogen = output_to_string('nitrogen.py')
nitrogen = nitrogen[nitrogen.find('Enthalpy'):]
with open('nitrogen.txt', 'w') as f:
    f.write(nitrogen)
ethane = output_to_string('ethane.py')
ethane = ethane[ethane.find('Internal'):]
with open('ethane.txt', 'w') as f:
    f.write(ethane)
gold = output_to_string('gold.py')
gold = gold[gold.find('Internal'):]
with open('gold.txt', 'w') as f:
    f.write(gold)

# Clean up, to not confuse git.
shutil.rmtree('vib')
shutil.rmtree('phonon')
