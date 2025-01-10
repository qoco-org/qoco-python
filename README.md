# QOCO Python
<p align="center">
  <img src="https://github.com/user-attachments/assets/7bd44fa7-d198-4739-bb79-a5c15e04a8de" alt="drawing" width="500"/>
</p>

<p align="center">
  <a href=https://github.com/qoco-org/qoco-python/actions/workflows/unit_tests.yml/badge.svg"><img src="https://github.com/qoco-org/qoco-python/actions/workflows/unit_tests.yml/badge.svg"/></a>
</p>

This repository contains the python wrapper for [QOCO](https://github.com/qoco-org/qoco) and the code generator QOCOGEN.

QOCOGEN is a code generator which takes in an second-order cone program problem family and generates a customized C solver (called qoco_custom) for the specified problem family which implements the same algorithm as QOCO. This customized solver is library-free, only uses static memory allocation, and can be a few times faster than QOCO.

## Bug reports

File any issues or bug reports using the [issue tracker](https://github.com/qoco-org/qoco-python/issues).

## Citing
```
 @misc{chari2025qoco,
    author = {Chari, Govind M. and Acikmese, Behcet},
    title = {QOCO},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/qoco-org/qoco}},
 }
```

## License
QOCO is licensed under the BSD-3-Clause license.
