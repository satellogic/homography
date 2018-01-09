homography
==========

[![Build Status](https://travis-ci.org/satellogic/homography.svg?branch=master)](https://travis-ci.org/satellogic/homography)
![Coverage Status](https://satellogic.github.io/homography/coverage.svg)

library for 2d homographies

The 3x3 homography transformation matrix for transformations in two
dimensions is illustrated below.

```
  | x0 |   | a  b  c | | x |
  | y0 | = | d  e  f | | y |
  | z0 |   | g  h  1 | | 1 |

x' = x0 / z0
y' = y0 / z0
```
