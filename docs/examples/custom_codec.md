# Custom codec

Run a program qccodec doesn't support by defining its codec in your own script. The
codec turns a `ProgramInput` into the program's command and files, and parses the
output back into qcdata. Pass it as `codec=`, or register it to use the program by name,
including as a geomeTRIC gradient child. qccodec's
[Writing a codec](https://qccodec.docs.afh.org/guides/writing-a-codec/)
guide covers parsers, failure detection, and testing.

```python
{!examples/custom_codec.py!}
```
