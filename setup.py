#!/usr/bin/env python3
import os.path
from setuptools import setup

version = '0.1.0'

setup(
    name='homography',
    version=version,
    author='Slava Kerner, Amit Aronovitch',
    author_email='amit@satellogic.com',
    description="""\
This is a library for dealing with 2d homography transformations.
""",
    long_description=open('README.md').read(),
    py_modules = ['homography'],
    license='GPLv3',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],
    install_requires=[
        'Shapely', 'numpy', 'affine'
    ],
    setup_requires=[
        'pytest-runner'
    ],
    tests_require=[
        'pytest'
    ],
    extras_require={
        'full': [
            'opencv-contrib-python',
        ]
    }
)
