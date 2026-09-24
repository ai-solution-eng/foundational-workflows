#!/usr/bin/env python3
"""Upload verification test data for the mm-rag watched-sources S3 sync.

Creates bucket mlis-models (if missing) and uploads:
  s3://mlis-models/showcase-docs/    3 markdown/text docs
  s3://mlis-models/showcase-reports/ 1 markdown report + 1 generated PDF

Every file carries unique retrieval markers (ZEPHYR9, AURORA7, HELIODRIVE,
FR-2026-0917) so end-to-end success can be proven by searching the RAG
datasets for these tokens.

Credentials: MinIO root from the pcai-se cluster (memory, 2026-09).
Endpoint: external HTTPS with the AIE private root CA.
"""
import sys

sys.path.insert(0, "/home/andrew/Code/HPE/SQLhandler/.dev/pys3lib")

import boto3
from botocore.config import Config

ENDPOINT = "https://minio-api.pcai-se-ai-application.hst.rdlabs.hpecorp.net"
CA = "/home/andrew/.config/opencode/pcai-aie-root-ca.crt"
KEY = "admin-io"
SECRET = "MinIO$2K"
BUCKET = "mlis-models"

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=KEY,
    aws_secret_access_key=SECRET,
    verify=CA,
    config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 3}),
)

# ---------------------------------------------------------------- content --

DOCS = {
    "zephyr-9-quickstart.md": """# Zephyr-9 Field Router — Quickstart (ZEPHYR9)

The Zephyr-9 field router ships with a quantum flux regulator pre-calibrated
to 42.7 THz. Before first boot:

1. Seat the regulator cartridge until the interlock clicks twice.
2. Connect the AuroraLink uplink (PoE++, port 0).
3. Hold the recessed sync button for 6 seconds; the status ring turns cyan.

Firmware 2.4.1 fixes the cold-boot regression seen on units below serial
Z9-10400. Units at firmware 2.3.x must upgrade before enabling mesh mode.

Warranty: 36 months. Support portal tag: ZEPHYR9-QS-2026.
""",
    "aurora-7-install-guide.md": """# Aurora-7 Edge Appliance — Installation Guide (AURORA7)

The Aurora-7 mounts in a 1U half-rack. Minimum clearances: 25 mm front,
150 mm rear for the airflow shroud.

Electrical: dual 650 W PSUs, 200-240 V. The HelioDrive storage module
(hot-swap NVMe, 7.68 TB) slides into bay A2; bay A1 is reserved for the
boot device.

Known issue: on chassis built before week 34/2026 the HelioDrive carrier
latch may report a false open state after reseating — power-cycle once to
clear. Reference tag AURORA7-IG-2026.
""",
    "heliodrive-release-notes.txt": """HelioDrive NVMe Storage Module — Release Notes 1.8.0 (HELIODRIVE)

New in 1.8.0:
- Endurance telemetry exposed over Redfish at /redfish/v1/Storage/HelioDrive.
- Power-loss protection capacitor self-test now runs at every boot.
- Fixed: SMART log page 0xC0 truncated at 512 bytes on some hosts.

Compatibility: requires Aurora-7 firmware 2.4.1 or later; Zephyr-9 routers
do not host HelioDrive modules (mechanical keying differs).
Tag: HELIODRIVE-RN-180.
""",
}

REPORTS_MD = {
    "2026-09-field-report.md": """# Field Service Report FR-2026-0917 (ZEPHYR9 / AURORA7)

Site: Bremerton substation 4. Engineer: on-call rotation B.

Summary: Two Zephyr-9 field routers reported intermittent AuroraLink drops
every ~40 minutes. Root cause: the quantum flux regulator on unit Z9-10388
had drifted below the 42.0 THz floor after a lightning transient; the pair
negotiated down and flapped.

Resolution: replaced regulator cartridge (spare part RF-4421), reseated
AuroraLink, verified 42.7 THz nominal on both units. Also updated the
adjacent Aurora-7 appliance to firmware 2.4.1 while on site (see
AURORA7-IG-2026 for the latch caveat).

Follow-ups: none open. Related: HELIODRIVE-RN-180 (no action required).
""",
}


def make_pdf(title, lines):
    """Hand-rolled minimal one-page PDF (Helvetica) with correct xref offsets."""
    def esc(s):
        return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

    parts = ["BT"]
    y = 760
    for i, ln in enumerate(lines):
        size = 18 if i == 0 else 11
        parts.append(f"/F1 {size} Tf 1 0 0 1 60 {y} Tm ({esc(ln)}) Tj")
        y -= 24 if i == 0 else 15
    parts.append("ET")
    stream = "\n".join(parts).encode("ascii")

    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return out


REPORT_PDF_NAME = "2026-09-capacity-review.pdf"
REPORT_PDF = make_pdf(
    "Capacity Review CR-2026-Q3",
    [
        "Capacity Review CR-2026-Q3 (AURORA7 fleet)",
        "",
        "Fleet: 214 Aurora-7 edge appliances across 18 sites.",
        "Mean HelioDrive utilisation: 61 percent. Headroom at site level",
        "is below 20 percent at Bremerton-4 and Millbrook-2; both are",
        "slated for a 7.68 TB HelioDrive swap in October 2026.",
        "Reference: FR-2026-0917 documents the Bremerton-4 visit.",
    ],
)

# ------------------------------------------------------------------ upload --

def main():
    # bucket create (idempotent)
    try:
        s3.head_bucket(Bucket=BUCKET)
        print(f"bucket {BUCKET}: exists")
    except s3.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchBucket"):
            s3.create_bucket(Bucket=BUCKET)
            print(f"bucket {BUCKET}: created")
        else:
            raise

    uploads = []
    for name, body in DOCS.items():
        uploads.append((f"showcase-docs/{name}", body.encode("utf-8"), "text/markdown"))
    for name, body in REPORTS_MD.items():
        uploads.append((f"showcase-reports/{name}", body.encode("utf-8"), "text/markdown"))
    uploads.append((f"showcase-reports/{REPORT_PDF_NAME}", REPORT_PDF, "application/pdf"))

    for key, data, ctype in uploads:
        s3.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=data,
            ContentType=ctype,
            Metadata={"source": "watched-sources-verify", "uploaded-by": "andrew-session"},
        )
        print(f"uploaded s3://{BUCKET}/{key}  ({len(data)} bytes, {ctype})")

    print("\n--- verify listing ---")
    for prefix in ("showcase-docs/", "showcase-reports/"):
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        print(f"s3://{BUCKET}/{prefix}: {resp['KeyCount']} object(s)")
        for o in resp.get("Contents", []):
            print(f"  {o['Key']}  {o['Size']}B  etag={o['ETag']}")


if __name__ == "__main__":
    main()
