import io
import os
import re

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

requirements = [
    'numpy',
    'pandas',
    'scipy',
    'NREL-PySAM',
    # 'dash',
]


# Read the version from the __init__.py file without importing it
def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(name='dwelling_model',
      version=find_version('dwelling_model', '__init__.py'),
      description='Residential building model for co-simulation',
      author='Killian McKenna',
      author_email='Killian.McKenna@nrel.gov',
      url='https://github.nrel.gov/Customer-Modeling/Dwelling_Object_Oriented_Model__DOOM',
      packages=['dwelling_model'],
      install_requires=requirements,
      package_data={'dwelling_model': [],
                    },
      )
