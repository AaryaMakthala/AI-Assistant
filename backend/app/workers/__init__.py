"""Worker package.

Importing the task modules here ensures the ``@celery_app.task`` decorators
execute when the package is loaded — which is what ``autodiscover_tasks`` does
under the hood.  Without these imports the worker process never sees
``ingestion.py`` or ``maintenance.py`` (it only looks for ``tasks.py``), and
every dispatched message is discarded as an unregistered task.
"""

from app.workers.ingestion import ingest_document  # noqa: F401
from app.workers.maintenance import reap_stuck_documents  # noqa: F401
