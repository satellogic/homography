#!/usr/bin/env python
from os import makedirs
from subprocess import check_call
from setup import __version__


def minor_from_full(version):
    return '.'.join(version.split('.')[:2])


makedirs('docs', exist_ok=True)
check_call('coverage-badge -f -o docs/coverage.svg'.split())
check_call(['sphinx-build',
            '-D', 'version={}'.format(minor_from_full(__version__)),
            '-D', 'release={}'.format(__version__),
            'sphinx', 'docs'])
