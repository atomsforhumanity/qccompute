# File Input

A `qcdata.FileInput` is a command line and the files it needs: an escape hatch for a
program qccodec does not support yet, or a calculation its codec does not cover (an MD
run, a script). qccompute runs it under qccodec's `files` program
(`pip install qccompute[qccodec]`) in a scratch directory and returns every file the
command wrote in `results.files`, with its output in `logs`.

```python
{!examples/file_input.py!}
```
