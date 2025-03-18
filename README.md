# QOCO Python
<p align="center">
  <img src="https://github.com/user-attachments/assets/7bd44fa7-d198-4739-bb79-a5c15e04a8de" alt="drawing" width="500"/>
</p>

<p align="center">
  <a href=https://github.com/qoco-org/qoco-python/actions/workflows/unit_tests.yml/badge.svg"><img src="https://github.com/qoco-org/qoco-python/actions/workflows/unit_tests.yml/badge.svg"/></a>
  <a href="https://img.shields.io/pypi/dm/qoco.svg?label=Pypi%20downloads"><img src="https://img.shields.io/pypi/dm/qoco.svg?label=Pypi%20downloads" alt="PyPI Downloads" /></a>
  <a href="https://qoco-org.github.io/qoco/"><img src="https://img.shields.io/badge/docs-online-brightgreen?logo=read-the-docs&style=flat" alt="Documentation" /></a>
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD_3--Clause-green.svg" alt="License" /></a>
</p>

This repository contains the python wrapper for [QOCO](https://github.com/qoco-org/qoco) and the code generator QOCOGEN.

QOCOGEN is a code generator which takes in an second-order cone program problem family and generates a customized C solver (called qoco_custom) for the specified problem family which implements the same algorithm as QOCO. This customized solver is library-free, only uses static memory allocation, and can be a few times faster than QOCO.

## Installation and Usage

You can install `qoco-python` by running `pip install qoco`.

For instructions on using the wrapper for QOCO, refer to [API](https://qoco-org.github.io/qoco/api/matlab.html#matlab-interface), and [simple example](https://qoco-org.github.io/qoco/examples/simple_example.html#simple-example) for an example of solving a simple SOCP with the python wrapper.

For instructions on using QOCOGEN, refer to the [documentation](https://qoco-org.github.io/qoco/codegen/index.html).

## Tests
To run tests, first install cvxpy and pytest
```bash
pip install cvxpy pytest
```

and execute:

```bash
pytest
```

## Bug reports

File any issues or bug reports using the [issue tracker](https://github.com/qoco-org/qoco-python/issues).

## Citing
```
@misc{chari2025qoco,
  title         = {Custom Solver Generation for Quadratic Objective Second-Order Cone Programs},
  author        = {Chari, Govind M and A{\c{c}}{\i}kme{\c{s}}e, Beh{\c{c}}et},
  year          = {2025},
  eprint        = {2503.12658},
  archiveprefix = {arXiv},
  primaryclass  = {math.OC},
  url           = {https://arxiv.org/abs/2503.12658}
}
```

## License
QOCO is licensed under the BSD-3-Clause license.
