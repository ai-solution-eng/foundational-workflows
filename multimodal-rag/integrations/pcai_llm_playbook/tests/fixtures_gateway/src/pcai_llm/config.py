"""MINIATURE fixture: settings stub so the miniature imports stand alone.

NOT the real gateway config - only the two attributes the miniature
memory_write module reads.
"""


class _MiniSettings:
    distiller_min_doc_chars = 0
    distiller_dedup_superseded = False


settings = _MiniSettings()
