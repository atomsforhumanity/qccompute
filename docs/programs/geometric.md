::: qccompute.adapters.geometric.GeometricAdapter
## Recursive program input

```python
from qcdata import ProgramInput, ProgramSpec, Structure
from qccompute import compute

structure = Structure.open("molecule.xyz")
input_data = ProgramInput(
    program="geometric",
    calctype="optimization",
    structure=structure,
    keywords={"maxiter": 250},
    subprograms=[
        ProgramSpec(
            program="terachem",
            calctype="gradient",
            model={"method": "wb97x-d3", "basis": "def2-svp"},
        ),
    ],
)
output = compute(input_data)
output.results.final_structure
output.results.trajectory[0].results.provenance  # Gradient producer
output.execution.wall_time  # Optimizer execution time
```

Each child has its own model, keywords, native files, command-line arguments, and
metadata. Additional role-named structures are passed to child calculations.
