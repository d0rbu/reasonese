"""Restore a verified task archive without traversing archive symlinks."""
import os
import shutil
import stat
import sys
import tarfile
from pathlib import Path, PurePosixPath

allowed = {"workspace", "python-env", "node-env", "go-modules", "tool-state"}
archive, destination = sys.argv[1:]
root = Path(destination)
root.mkdir(parents=True, exist_ok=False)


def path(name):
    parts = PurePosixPath(name).parts
    if not parts or parts[0] not in allowed or any(p in {"..", "."} for p in parts) or name.startswith("/"):
        raise ValueError(f"Unsafe archive path: {name!r}")
    result = root.joinpath(*parts)
    for parent in result.parents:
        if parent == root:
            break
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise ValueError(f"Unsafe archive parent: {name!r}")
    return result


with tarfile.open(archive, "r:gz") as source:
    seen = set()
    links = []
    directories = []
    for member in source:
        if member.name in seen:
            raise ValueError(f"Duplicate archive path: {member.name!r}")
        seen.add(member.name)
        target = path(member.name)
        if len(PurePosixPath(member.name).parts) == 1 and not member.isdir():
            raise ValueError("Task roots must be directories")
        target.parent.mkdir(parents=True, exist_ok=True)
        if member.isdir():
            target.mkdir(exist_ok=True)
            directories.append((target, member))
        elif member.isfile():
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError("Missing regular-file payload")
            with target.open("xb") as output:
                shutil.copyfileobj(stream, output)
            os.chmod(target, member.mode & 0o777)
            os.utime(target, (member.mtime, member.mtime))
        elif member.issym() or member.islnk():
            links.append(member)
        else:
            raise ValueError(f"Unsupported archive member: {member.name!r}")
    for member in links:
        if not member.islnk():
            continue
        target, origin = path(member.name), path(member.linkname)
        if not stat.S_ISREG(origin.lstat().st_mode):
            raise ValueError("Hard links must target an extracted regular file")
        os.link(origin, target, follow_symlinks=False)
    for member in links:
        if member.issym():
            # Store literal targets, including neutral absolute mount paths.
            # Do not dereference them during extraction on the host.
            path(member.name).symlink_to(member.linkname)
    for target, member in reversed(directories):
        os.chmod(target, member.mode & 0o777)
        os.utime(target, (member.mtime, member.mtime))
