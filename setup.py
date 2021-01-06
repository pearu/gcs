import os
import sys
import builtins
import versioneer

if sys.version_info[:2] < (3, 4):
    raise RuntimeError("Python version >= 3.4 required.")

builtins.__GCS_SETUP__ = True

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

CONDA_BUILD = int(os.environ.get('CONDA_BUILD', '0'))

from setuptools import setup, find_packages  # noqa: E402

DESCRIPTION = "GCS - Generalized Compressed Storage of multidimensional arrays"
LONG_DESCRIPTION = """
The aim of the GCS project is to provide a prototype of
Generalized Compressed Storage of multidimensional arrays using
dimensionality reduction approach.  """


def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    if CONDA_BUILD:
        # conda dependencies are specified in meta.yaml
        install_requires = []
        setup_requires = []
        tests_require = []
    else:
        install_requires = open('requirements.txt', 'r').read().splitlines()
        setup_requires = ['pytest-runner']
        tests_require = ['pytest']

    metadata = dict(
        name='gcs',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license='BSD',
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        author='Pearu Peterson',
        maintainer='Pearu Peterson',
        author_email='pearu.peterson@gmail.com',
        url='https://github.com/pearu/gcs',
        platforms='Cross Platform',
        classifiers=[
            "Intended Audience :: Developers",
            "License :: OSI Approved :: BSD License",
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            "Operating System :: OS Independent",
            "Topic :: Software Development",
        ],
        packages=find_packages(),
        install_requires=install_requires,
        setup_requires=setup_requires,
        tests_require=tests_require,
    )

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)
    return


if __name__ == '__main__':
    setup_package()
    del builtins.__GCS_SETUP__
