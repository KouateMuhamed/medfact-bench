"""Pure endpoint-backed chat harness for MedFact-Bench."""

from verifiers.v1.harnesses.null import NullHarness, NullHarnessConfig

MedFactHarnessConfig = NullHarnessConfig


class MedFactHarness(NullHarness):
    """Run the benchmark prompt without adding instructions or tools."""
