"""Linux abstract-socket multiprocessing IPC for the training loader.

Shared filesystems may not support filesystem AF_UNIX sockets.
Linux abstract sockets live only in the kernel and still use the standard
multiprocessing authentication and file-descriptor transfer implementation.
"""
import multiprocessing.connection as connection
import os
import sys
import uuid

_original_arbitrary_address = connection.arbitrary_address

def _ipc_address(family):
    if family=="AF_UNIX":
        return "\0ditgs-view-"+str(os.getpid())+"-"+uuid.uuid4().hex
    return _original_arbitrary_address(family)

def configure_loader_ipc():
    """Idempotent, Linux-only; does not touch TMPDIR or tensor sharing strategy."""
    if sys.platform.startswith("linux"):
        connection.arbitrary_address=_ipc_address

def configure_loader_worker(worker_id):
    # Explicit worker hook also handles spawn/forkserver, not just inherited fork.
    configure_loader_ipc()
