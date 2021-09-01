#!/usr/bin/env python
import os

from setuptools import setup, find_packages


if __name__ == "__main__":

    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, "src")

    about = {}
    with open(os.path.join(src_dir, "vivarium_ciff_sam", "__about__.py")) as f:
        exec(f.read(), about)

    with open(os.path.join(base_dir, "README.rst")) as f:
        long_description = f.read()

    install_requirements = [
        'vivarium==0.10.9',
        'vivarium_public_health==0.10.13',
        'click',
        'gbd_mapping>=3.0.0, <4.0.0',
        'jinja2',
        'loguru',
        'numpy',
        'pandas',
        'scipy',
        'tables',
        'pyyaml',
    ]

    # use "pip install -e .[dev]" to install required components + extra components
    internal = [
        'vivarium_inputs[data]==4.0.4',
        'vivarium_cluster_tools>=1.2.6',
    ]

    interactive = [
        'jupyterlab',
        'matplotlib',
        'sympy',
    ]

    setup(
        name=about['__title__'],
        version=about['__version__'],

        description=about['__summary__'],
        long_description=long_description,
        license=about['__license__'],
        url=about["__uri__"],

        author=about["__author__"],
        author_email=about["__email__"],

        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        include_package_data=True,

        install_requires=install_requirements,
        extras_require={
            'interactive': interactive,
            'dev': internal + interactive,
        },

        zip_safe=False,

        entry_points='''
            [console_scripts]
            make_artifacts=vivarium_ciff_sam.tools.cli:make_artifacts
            make_results=vivarium_ciff_sam.tools.cli:make_results
        '''
    )
