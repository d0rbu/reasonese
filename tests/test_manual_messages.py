from __future__ import annotations

from pathlib import Path

import pytest

from reasonese.axes import Assistant, Author, Channel, Framing, Instruction
from reasonese.conversation import GeneratedMessage, GeneratedText, construct_conversation
from reasonese.manual_messages import ManualMessageLibrary
from reasonese.matchup import make_matchup
from reasonese.planning import PromptSpec


def _spec(
    text: str = "Do the task.",
    framing: Framing = Framing.NORMAL,
    channel: Channel = Channel.USER,
    author: Author = Author.USER,
) -> PromptSpec:
    return PromptSpec(Instruction.parse(text), framing, channel, author)


def _write_instruction(
    root: Path,
    name: str = "task",
    instruction: str = "Do the task.",
    variants: dict[Framing, str] | None = None,
) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "instruction.txt").write_text(instruction)
    variants = variants or {}
    for framing in Framing:
        (directory / f"{framing}.txt").write_text(
            variants.get(framing, f"manual {framing} version")
        )
    return directory


def test_manual_library_selects_instruction_and_framing_independently_of_channel(
    tmp_path: Path,
) -> None:
    _write_instruction(
        tmp_path,
        variants={
            Framing.NORMAL: "Please do the task.",
            Framing.CASUAL: "hey can u do the task",
        },
    )
    library = ManualMessageLibrary(tmp_path)

    assert library.message_for(_spec(framing=Framing.NORMAL)) == "Please do the task."
    assert (
        library.message_for(_spec(framing=Framing.CASUAL, channel=Channel.README))
        == "hey can u do the task"
    )


def test_manual_library_rejects_placeholders_non_user_authors_and_missing_instructions(
    tmp_path: Path,
) -> None:
    _write_instruction(tmp_path, variants={Framing.NORMAL: "TODO: write this"})
    library = ManualMessageLibrary(tmp_path)

    with pytest.raises(ValueError, match="still a placeholder"):
        library.message_for(_spec())
    with pytest.raises(ValueError, match="only defined for the user"):
        library.message_for(_spec(author=Author.INKLING))
    with pytest.raises(ValueError, match="no manual message directory"):
        library.message_for(_spec("Different task."))


def test_manual_library_validates_root_complete_trees_and_unique_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        ManualMessageLibrary(tmp_path / "missing").message_for(_spec())

    incomplete = tmp_path / "incomplete"
    directory = incomplete / "task"
    directory.mkdir(parents=True)
    (directory / "instruction.txt").write_text("Do the task.")
    with pytest.raises(ValueError, match="is missing"):
        ManualMessageLibrary(incomplete).message_for(_spec())

    duplicate = tmp_path / "duplicate"
    _write_instruction(duplicate, "first")
    _write_instruction(duplicate, "second")
    with pytest.raises(ValueError, match="appears more than once"):
        ManualMessageLibrary(duplicate).message_for(_spec())


def test_manual_library_matches_cached_setup_and_detects_edits(tmp_path: Path) -> None:
    directory = _write_instruction(
        tmp_path,
        variants={Framing.NORMAL: "Manual first request."},
    )
    user = _spec()
    model = _spec("Other task.", author=Author.INKLING)
    matchup = make_matchup((model, user), Assistant.INKLING_SMALL)
    library = ManualMessageLibrary(tmp_path)
    setup = construct_conversation(
        matchup,
        (
            GeneratedMessage(model, GeneratedText.parse("Model request."), None),
            GeneratedMessage(user, library.message_for(user), None),
        ),
    )

    assert library.matches(setup) is True
    (directory / "normal.txt").write_text("Manual changed request.")
    assert library.matches(setup) is False


def test_manual_snapshot_is_stable_but_next_snapshot_detects_edits(tmp_path: Path) -> None:
    directory = _write_instruction(
        tmp_path,
        variants={Framing.NORMAL: "Snapshot request."},
    )
    spec = _spec()
    library = ManualMessageLibrary(tmp_path)
    snapshot = library.snapshot((spec,))

    (directory / "normal.txt").write_text("Changed after snapshot.")

    assert snapshot.message_for(spec) == "Snapshot request."
    assert library.snapshot((spec,)).message_for(spec) == "Changed after snapshot."


def test_model_only_snapshot_does_not_require_manual_directory(tmp_path: Path) -> None:
    spec = _spec(author=Author.INKLING)
    snapshot = ManualMessageLibrary(tmp_path / "missing").snapshot((spec,))

    assert not snapshot.variants
    with pytest.raises(ValueError, match="only defined for the user"):
        snapshot.message_for(spec)


def test_repository_contains_placeholder_tree_for_every_example_instruction() -> None:
    root = Path("prompts/user")
    directories = tuple(path for path in root.iterdir() if path.is_dir())
    assert len(directories) == 4
    expected = {"instruction.txt", *(f"{framing}.txt" for framing in Framing)}
    for directory in directories:
        assert {path.name for path in directory.iterdir() if path.is_file()} == expected
        for framing in Framing:
            assert (directory / f"{framing}.txt").read_text().startswith("TODO:")
