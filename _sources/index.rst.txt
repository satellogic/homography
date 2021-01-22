|Build Status|_ |Coverage Status|

.. |Build Status| image:: https://travis-ci.org/satellogic/homography.svg?branch=master
	          :alt: Build Status
.. _Build Status: https://travis-ci.org/satellogic/homography

.. |Coverage Status| image:: https://satellogic.github.io/homography/coverage.svg
                     :alt: Coverage Status

=============================
Homography package for Python
=============================

A 2D homography from :math:`\left(\begin{smallmatrix}x \\ y\end{smallmatrix}\right)` to
:math:`\left(\begin{smallmatrix}x' \\ y'\end{smallmatrix}\right)`
can be represented by 8 parameters :math:`(a,b,\ldots h)`, organized in a 3x3 matrix.
The transformation rule is as follows:

.. math::

   \begin{pmatrix}
     x_0 \\ y_0 \\ z_0
   \end{pmatrix} &=
       \begin{pmatrix}
           a & b & c \\
           d & e & f \\
           g & h & 1
       \end{pmatrix}
           \begin{pmatrix}
              x \\ y \\ 1
           \end{pmatrix} \\
   \begin{pmatrix}
     x' \\ y'
   \end{pmatrix} &= \frac{1}{z_0}
        \begin{pmatrix}
           x_0 \\ y_0
        \end{pmatrix}

``homography`` module
=====================

.. automodule:: homography
   :members:
   :undoc-members:
   :special-members:
   :exclude-members: __dict__, __str__, __repr__, __hash__

.. toctree::
   :maxdepth: 2
