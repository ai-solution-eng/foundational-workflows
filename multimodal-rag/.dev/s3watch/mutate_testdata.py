#!/usr/bin/env python3
"""Mutate the watched-sources test data to exercise the sync's diff engine.

For the next tick (2026-09-24 09:45):
  * zephyr-9-quickstart.md  -> UPDATED (new ETag/Size => must re-ingest)
  * heliodrive-release-notes.txt -> DELETED upstream (=> must be pruned)
  * aurora-7-install-guide.md    -> untouched (=> must be ETag-skipped)
  * showcase-reports/*           -> untouched (=> must be ETag-skipped)
"""
import sys

sys.path.insert(0, "/home/andrew/Code/HPE/SQLhandler/.dev/pys3lib")

import boto3
from botocore.config import Config

ENDPOINT = "https://minio-api.pcai-se-ai-application.hst.rdlabs.hpecorp.net"
CA = "/home/andrew/.config/opencode/pcai-aie-root-ca.crt"

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id="admin-io",
    aws_secret_access_key="MinIO$2K",
    verify=CA,
    config=Config(s3={"addressing_style": "path"}),
)

updated = """# Zephyr-9 Field Router — Quickstart (ZEPHYR9)

The Zephyr-9 field router ships with a quantum flux regulator pre-calibrated
to 42.7 THz. Before first boot:

1. Seat the regulator cartridge until the interlock clicks twice.
2. Connect the AuroraLink uplink (PoE++, port 0).
3. Hold the recessed sync button for 6 seconds; the status ring turns cyan.

Firmware 2.4.1 fixes the cold-boot regression seen on units below serial
Z9-10400. Units at firmware 2.3.x must upgrade before enabling mesh mode.

Field Notice FN-2026-0924 (ZEPHYR9): regulator cartridges from lot RF-4421-B
must be recalibrated after 500 thermal cycles; the field-service tool
reports cycle count during every boot self-test.

Warranty: 36 months. Support portal tag: ZEPHYR9-QS-2026.
"""

s3.put_object(
    Bucket="mlis-models",
    Key="showcase-docs/zephyr-9-quickstart.md",
    Body=updated.encode("utf-8"),
    ContentType="text/markdown",
    Metadata={"source": "watched-sources-verify", "revision": "2"},
)
print("updated s3://mlis-models/showcase-docs/zephyr-9-quickstart.md (rev 2, adds FN-2026-0924)")

s3.delete_object(Bucket="mlis-models", Key="showcase-docs/heliodrive-release-notes.txt")
print("deleted s3://mlis-models/showcase-docs/heliodrive-release-notes.txt")

for prefix in ("showcase-docs/", "showcase-reports/"):
    resp = s3.list_objects_v2(Bucket="mlis-models", Prefix=prefix)
    print(f"s3://mlis-models/{prefix}: {resp['KeyCount']} object(s)")
    for o in resp.get("Contents", []):
        print(f"  {o['Key']}  {o['Size']}B")
