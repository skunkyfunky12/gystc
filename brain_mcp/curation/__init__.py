"""Vault curation — the write/maintenance layer on the shared brain.

The vault is irreplaceable and NOT in git by default, so every module here treats
reversibility (vault_git) as the prerequisite, proposes before applying, archives
instead of deleting, and commits one revertable snapshot per run.
"""
