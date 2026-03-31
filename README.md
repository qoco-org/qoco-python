# QOCO Python
<p align="center">
  <img src="https://github.com/user-attachments/assets/7bd44fa7-d198-4739-bb79-a5c15e04a8de" alt="drawing" width="500"/>
</p>

<p align="center">
  <a href=https://github.com/qoco-org/qoco-python/actions/workflows/unit_tests.yml/badge.svg"><img src="https://github.com/qoco-org/qoco-python/actions/workflows/unit_tests.yml/badge.svg"/></a>
  <a href="https://img.shields.io/pypi/dm/qoco.svg?label=Pypi%20downloads"><img src="https://img.shields.io/pypi/dm/qoco.svg?label=Pypi%20downloads" alt="PyPI Downloads" /></a>
  <a href="https://arxiv.org/abs/2503.12658"><img src="http://img.shields.io/badge/arXiv-2503.12658-B31B1B.svg"/></a>
  <a href="https://qoco-org.github.io/qoco/"><img src="https://img.shields.io/badge/docs-online-brightgreen?logo=read-the-docs&style=flat" alt="Documentation" /></a>
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD_3--Clause-green.svg" alt="License" /></a>
</p>

This repository contains the python wrapper for [QOCO](https://github.com/qoco-org/qoco).


## Installation and Usage

You can install `qoco-python` by running `pip install qoco`.

For instructions on using the wrapper for QOCO, refer to [API](https://qoco-org.github.io/qoco/api/matlab.html#matlab-interface), and [simple example](https://qoco-org.github.io/qoco/examples/simple_example.html#simple-example) for an example of solving a simple SOCP with the python wrapper.

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
@article{chari2026qoco,
  title = {QOCO: a quadratic objective conic optimizer with custom solver generation},
  author = {Chari, Govind M. and A\c{c}ıkmeşe, Beh\c{c}et},
  journal = {Mathematical Programming Computation},
  issn = {1867-2957},
  url = {http://dx.doi.org/10.1007/s12532-026-00311-8},
  doi = {10.1007/s12532-026-00311-8},
  publisher = {Springer Science and Business Media LLC},
  year = {2026},
  month = mar,
}
```

## License
QOCO is licensed under the BSD-3-Clause license.
