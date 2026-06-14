"""Commentary-driven observability layer for DVSA.

A parallel, orthogonal telemetry layer that turns vision-routine output into
wide, query-time-aggregatable commentary events (see ``schema``, ``commentator``,
``sinks`` and ``aggregation``). Importing this package pulls in only the
stdlib-only core; the Django model/API shell lives in ``models``/``views`` and is
loaded by Django via ``apps.ObservabilityConfig``.
"""
