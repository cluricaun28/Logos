---
name: codebase-backup
description: Versioned backup system for project directories with git-based snapshots and rsync mirrors.
trigger_keywords: [backup, snapshot, versioned, archive, mirror, rsync, preserve]
category: devops
created: YYYY-MM-DD
last_updated: YYYY-MM-DD
confidence: high
---

# Codebase Backup Skill

## Trigger Conditions

Use this skill when:
- Backing up a project directory before making destructive changes
- Creating versioned snapshots of work-in-progress
- Mirroring a codebase to an external location (NAS, Windows host, remote server)
- Preserving state before running migrations or updates

## Prerequisites

- `git` installed and available in PATH
- Target backup destination accessible (local path, mounted drive, or SSH-accessible server)
- Sufficient disk space at destination

## Steps

### 1. Git-Based Snapshot (If Repository Exists)

```bash
cd /path/to/project
git add -A
git commit -m "backup: snapshot YYYY-MM-DD HH:MM"
git tag -a "backup-YYYY-MM-DD" -m "Automated backup before [reason]"
```

### 2. Rsync Mirror (For Non-Git Directories or External Backup)

```bash
# Local mirror
rsync -avz --delete \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  /path/to/source/ \
  /path/to/backup/destination/

# Remote mirror (via SSH)
rsync -avz -e ssh \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  /path/to/source/ \
  user@remote:/path/to/backup/
```

### 3. WSL-to-Windows Backup (Common Pattern)

```bash
# Backup from WSL to Windows Desktop
rsync -avz \
  --exclude='.git/' \
  ~/.hermes/ \
  /mnt/c/Users/USERNAME/Desktop/hermes-backup-$(date +%Y%m%d)/
```

Replace `USERNAME` with the actual Windows username. Find it with:
```bash
ls /mnt/c/Users/
```

## Pitfalls

- **Don't backup `.git/` directories** — they're large and already versioned. Use git tags instead.
- **WSL filesystem is slow for Windows-side access** — keep backups in WSL ext4 when possible, only rsync to `/mnt/c/` for cross-platform safety.
- **`--delete` flag removes files at destination that don't exist at source** — omit if you want additive-only backups.
- **Check disk space before large rsync operations** — use `df -h` on the target.

## Verification

After backup:
```bash
# Verify file count matches
find /path/to/source/ -type f | wc -l
find /path/to/backup/ -type f | wc -l

# Verify recent files exist
ls -la /path/to/backup/latest-file-or-directory/
```

## Related Skills

- `wsl-backup-to-windows` — WSL-specific backup patterns
- `docker-compose-image-troubleshooting` — if backing up Docker environments
