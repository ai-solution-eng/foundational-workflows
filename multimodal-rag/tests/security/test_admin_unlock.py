"""Admin unlock path: an admin must PROVE a dataset password, never bypass it.

Live defect (G2, found 2026-10-14): an admin session could not download a
password-protected dataset at all.  _annotate_acl_granted returned early for
admins, so the /access page never stamped acl_granted and rendered the only
password affordance as "Select..." -- which admins cannot use
(_require_select_identity 409s: "Dataset selection applies to per-user API
keys").  The read then fell through to 401 "is password protected" with no
way to unlock.

The fix serves an admin's POST /select as an UNLOCK: verify the password,
warm the unlock cache, report source="admin-unlock".  The security property
these tests pin is that NO bypass was introduced:

  * admin + correct password  -> 200, cache warmed
  * admin + WRONG password    -> 403 (never a silent success)
  * admin + missing password  -> 400 (never a silent success)
  * non-admin behaviour is unchanged (still goes through _require_select_identity)
  * admin rows render as acl_granted (accurate: dataset_allowed is True for admins)

Run:

    pytest tests/security/test_admin_unlock.py -q
"""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.api_server as api


def test_annotation_stamps_admin_granted():
    """Admins render as granted so the panel offers the password form."""
    src = inspect.getsource(api._annotate_acl_granted)
    assert "identity.is_admin" in src, "admin branch missing"
    assert 'ds["acl_granted"] = True' in src, "admin rows must be stamped granted"
    # The old early-return shape must be gone.
    assert "identity is None or identity.is_admin:" not in src, (
        "the early-return-for-admin shape is back — admins would lose the "
        "password affordance again"
    )


def test_select_serves_admin_as_unlock_without_bypass():
    """The admin branch must VERIFY the password; no silent success path."""
    src = inspect.getsource(api.api_select_dataset)
    assert "admin-unlock" in src, "admin unlock branch missing"
    # The branch must verify and must be able to refuse.
    assert "verify_password" in src, "admin branch must verify the password"
    assert "Incorrect password" in src, "admin branch must refuse a wrong password"
    assert '"password" is required' in src or "password' is required" in src, (
        "admin branch must require a password rather than defaulting to success"
    )


def test_admin_branch_precedes_select_identity_gate():
    """The admin branch must run BEFORE _require_select_identity (which 409s)."""
    src = inspect.getsource(api.api_select_dataset)
    admin_at = src.find("admin-unlock")
    select_gate_at = src.find("_require_select_identity()")
    assert admin_at != -1, "admin branch missing"
    assert select_gate_at != -1, "_require_select_identity call missing"
    assert admin_at < select_gate_at, (
        "the admin branch must be evaluated before the select-identity gate, "
        "or admins still get the 409 instead of a password prompt"
    )


def test_no_admin_bypass_in_the_password_gate():
    """REGRESSION GUARD: _require_dataset_password must NOT gain an admin bypass.

    The password protects the USER's data from the operator; an admin must
    prove it like anyone else.  If someone "fixes" the download 401 by
    short-circuiting this gate for admins, that is a security regression and
    this test must fail.
    """
    src = inspect.getsource(api._require_dataset_password)
    assert "is_admin" not in src, (
        "_require_dataset_password gained an admin bypass — the dataset "
        "password must still be proven by admins"
    )


def test_admin_unlock_branch_is_narrowed_to_protected_datasets():
    """The admin branch must only engage for PASSWORD-PROTECTED datasets.

    An admin has nothing to select INTO, so the historical 409 stays for an
    unprotected dataset (pinned by
    tests/security/test_access_store.py::test_rest_selection_admin_key_refused).
    The branch exists solely to obtain a password proof.
    """
    src = inspect.getsource(api.api_select_dataset)
    admin_at = src.find("admin-unlock")
    assert admin_at != -1
    branch = src[:admin_at]
    assert "has_password" in branch, (
        "the admin branch must gate on has_password, or it changes the "
        "historical 409 for unprotected datasets"
    )
    assert "409" in branch, "the non-protected case must keep the historical 409"
