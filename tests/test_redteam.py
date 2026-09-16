from pathlib import Path

from ml_lab.redteam.service import RedTeamCase, RedTeamMutator, RedTeamService
from ml_lab.storage.workspace import Workspace


def test_redteam_suite_is_seeded_and_order_independent(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Red team")
    service = RedTeamService(workspace)

    cases = (
        RedTeamCase("case-b", {"text": "open the door"}, {"decision": "ASK"}),
        RedTeamCase("case-a", {"text": "hit the goblin"}, {"decision": "NO_ACTION"}),
    )

    def append_noise(payload: object, rng: object) -> object:
        assert isinstance(payload, dict)
        chooser = getattr(rng, "choice")
        return {**payload, "noise": chooser(["uh", "erm", "...", "pls"])}

    def case_flip(payload: object, rng: object) -> object:
        assert isinstance(payload, dict)
        chooser = getattr(rng, "choice")
        return {"text": str(payload["text"]).swapcase(), "suffix": chooser(["?", "!", "..."])}

    mutators = (
        RedTeamMutator("noise", append_noise),
        RedTeamMutator("case-flip", case_flip),
    )
    first = service.run_suite(
        project_id=project.id,
        seed=1234,
        mutator_version="test-v1",
        base_cases=cases,
        mutators=mutators,
    )
    second = service.run_suite(
        project_id=project.id,
        seed=1234,
        mutator_version="test-v1",
        base_cases=tuple(reversed(cases)),
        mutators=tuple(reversed(mutators)),
    )
    assert service.generated_cases(first.id) == service.generated_cases(second.id)


def test_different_redteam_seed_changes_generated_cases(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Red team")
    service = RedTeamService(workspace)
    case = RedTeamCase("case", {"text": "look around"}, {"decision": "ASK"})

    def mutate(payload: object, rng: object) -> object:
        assert isinstance(payload, dict)
        randint = getattr(rng, "randint")
        return {**payload, "nonce": randint(0, 1_000_000)}

    mutator = RedTeamMutator("nonce", mutate)
    first = service.run_suite(
        project_id=project.id,
        seed=1,
        mutator_version="v1",
        base_cases=(case,),
        mutators=(mutator,),
    )
    second = service.run_suite(
        project_id=project.id,
        seed=2,
        mutator_version="v1",
        base_cases=(case,),
        mutators=(mutator,),
    )
    assert service.generated_cases(first.id) != service.generated_cases(second.id)
