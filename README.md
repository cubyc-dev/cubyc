[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cubyc)](https://pypi.org/project/cubyc/)
[![PyPI Status](https://badge.fury.io/py/cubyc.svg)](https://badge.fury.io/py/cubyc)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/cubyc)](https://pepy.tech/project/cubyc)
[![license](https://img.shields.io/badge/License-LGPL%203.0-blue.svg)](https://opensource.org/licenses/LGPL-3.0)


<div align="center"> 

<img alt="Lightning" src="img/banner.png" width="800px" style="max-width: 100%;">

<br/>
<br/>

**The repository for all your experiments**

</div>

---

<p align="center">
    <a href="#quickstart">Quickstart</a> •
    <a href="https://docs.cubyc.com">Documentation</a> •
    <a href="#contributing">Contributing</a> •
    <a href="#license">License</a> •
    <a href="#contact">Contact</a>
</p>

Cubyc is an open-source experiment tracking library for data scientists.
With Cubyc, you can easily track, version, and analyze your experiments using Git and SQL, all without ever leaving your
Python environment.

# Quickstart

Install Cubyc:

```bash
pip install cubyc
```

Initialize a new project in your current directory:

```bash
cubyc init
```

Start tracking your experiments:

```python
import numpy as np
from cubyc import Run

@Run(tags=["linear_algebra"])
def matrix_multiplication(n_size: int):
    A = np.random.rand(n_size, n_size)
    B = np.random.rand(n_size, n_size)

    _ = np.dot(A, B)

for n_size in [10, 20, 40, 80, 160, 320, 640]:
    matrix_multiplication(n_size=n_size)
```

Analyze your runs with SQL:

```python
from cubyc import query

statement = """
                SELECT config.n_size, metadata.runtime
                FROM config
                INNER JOIN metadata ON config.id = metadata.id
                ORDER BY metadata.runtime ASC
            """
            
print(query(statement=statement))
```

Output:

```console
>>>    n_size   runtime
... 0      10  0.012209
... 1      20  1.455673
... 2      40  2.768197
... 3      80  4.073367
... 4     160  5.336599
... 5     320  6.663631
... 6     640  8.028414
```

# Documentation

For more information and examples on how to use Cubyc, please refer to our [documentation](https://docs.cubyc.com).

# Contributing

We welcome contributions from the community! If you'd like to contribute to Cubyc, please read our contributing
guidelines and code of conduct.

# License

Cubyc is released under the [LGPL-3.0 License](https://opensource.org/licenses/LGPL-3.0).

# Contact

If you have any questions, feedback, or suggestions, please feel free to open an issue or join
our [community](docs.cubyc.com/getting_started/community).