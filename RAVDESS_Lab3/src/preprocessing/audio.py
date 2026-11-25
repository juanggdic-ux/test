"""Data ingestion and preprocessing for paired RAVDESS emotional speech."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
import zipfile

import numpy as np
import soundfile as sf

EMOTION_CODES = {"Neutral": "01", "Sad": "04"}
_RAVDESS_RE = re.compile(r"^(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)\.wav$", re.I)


def parse_filename(name: str) -> dict[str, str] | None:
    """Parse RAVDESS identifiers or return None for nonmatching filenames."""
    match = _RAVDESS_RE.fullmatch(Path(name).name)
    if match is None:
        return None
    return dict(zip(("modality", "channel", "emotion", "intensity", "statement", "repetition", "actor"), match.groups()))


def choose_matched_members(
    names: list[str],
    n_actors: int | None = 24,
    statement: str = "01",
    repetition: str = "01",
    intensity: str = "01",
) -> tuple[list[str], dict[str, list[str]]]:
    """Select one neutral and one sad speech WAV for each matched actor.

    Uses filename metadata, not the 60-files-per-actor archive-order assumption.
    The same statement, repetition, intensity, and actor are used in both groups.
    """
    found: dict[str, dict[str, str]] = {}
    for name in names:
        meta = parse_filename(name)
        if meta is None or meta["modality"] != "03" or meta["channel"] != "01":
            continue
        if (meta["statement"], meta["repetition"], meta["intensity"]) != (statement, repetition, intensity):
            continue
        for label, emotion_code in EMOTION_CODES.items():
            if meta["emotion"] == emotion_code:
                found.setdefault(meta["actor"], {})[label] = name

    actors = sorted(a for a, pairs in found.items() if all(k in pairs for k in EMOTION_CODES))
    if n_actors is not None:
        if n_actors < 1:
            raise ValueError("n_actors must be positive, or None for all matching actors")
        actors = actors[:n_actors]
    if not actors:
        raise ValueError("No paired neutral/sad speech records found; verify archive and RAVDESS filename codes")
    return actors, {condition: [found[actor][condition] for actor in actors] for condition in EMOTION_CODES}


def read_wav_in_memory(archive: zipfile.ZipFile, member: str) -> tuple[np.ndarray, int]:
    """Decode one ZIP-contained WAV without writing a temporary WAV to disk."""
    audio_bytes = archive.read(member)
    signal, fs = sf.read(BytesIO(audio_bytes), dtype="float32", always_2d=True)
    mono = signal.mean(axis=1)  # Voice is analysed in mono.
    return mono, int(fs)


def load_paired_audios(zip_path: str | Path, n_actors: int | None = 24) -> tuple[list[str], dict[str, list[np.ndarray]], int]:
    """Load matched neutral/sad recordings from a ZIP directly into memory.

    Returns actor ids, two lists of 1-D arrays, and a shared sample rate.
    """
    with zipfile.ZipFile(zip_path) as archive:
        actors, members = choose_matched_members(archive.namelist(), n_actors)
        groups: dict[str, list[np.ndarray]] = {}
        rates: set[int] = set()
        for condition, filenames in members.items():
            groups[condition] = []
            for filename in filenames:
                samples, fs = read_wav_in_memory(archive, filename)
                groups[condition].append(samples)
                rates.add(fs)
    if len(rates) != 1:
        raise ValueError("Audio sample rates differ; explicitly resample before comparison")
    return actors, groups, rates.pop()


def truncate_matched_groups(groups: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """Truncate all groups to one shared sample count, returning 2D matrices."""
    records = [audio for audios in groups.values() for audio in audios]
    if not records:
        raise ValueError("No recordings available")
    target_length = min(map(len, records))
    if target_length == 0:
        raise ValueError("Empty recording found")
    return {condition: np.stack([audio[:target_length] for audio in audios]) for condition, audios in groups.items()}


def load_members_via_disk(zip_path: str | Path, members: list[str], temp_dir: str | Path) -> list[np.ndarray]:
    """Reference ingestion implementation: extract each WAV before reading it."""
    with zipfile.ZipFile(zip_path) as archive:
        paths = [archive.extract(member, path=temp_dir) for member in members]
    return [sf.read(path, dtype="float32", always_2d=True)[0].mean(axis=1) for path in paths]
