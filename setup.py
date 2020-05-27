#!/usr/bin/env python3
from setuptools import setup

__version__ = '0.1.6'


if __name__ == '__main__':
    setup(
        name='homography',
        version=__version__,
        author='Slava Kerner, Amit Aronovitch',
        url='https://github.com/satellogic/homography',
        author_email='amit@satellogic.com',
        description=(
            'A library for dealing with 2d homography transformations.'),
        long_description=open('README.rst').read(),
        py_modules=['homography'],
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
            'numpy', 'affine'
        ],
        extras_require={
            'full': [
                'Shapely',
                'opencv-contrib-python',
            ]
        }
    )
