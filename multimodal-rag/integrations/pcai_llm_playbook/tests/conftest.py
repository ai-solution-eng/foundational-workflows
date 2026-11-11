"""Pytest configuration for the gateway playbook overlay tests.

The vendored tree under fixtures_gateway/ is DATA, not tests: it is a miniature copy of the
gateway modules that carry the patcher's anchors, used by test_playbook_emission.py to exercise
the anchor logic offline. Its tests/test_distiller.py is part of that miniature and imports
pcai_llm, which does not exist in this repo — collecting it fails the whole run.

Ignore the fixture tree here so "pytest integrations/pcai_llm_playbook/tests" works from the
repo root. The real gateway suites are run against a patched /tmp copy of the actual checkout;
see apply_playbook_patch.py --help and documentation/PLAYBOOK-MEMORY.md.
"""

collect_ignore_glob = ["fixtures_gateway/*"]
