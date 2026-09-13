from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import dflash2_tree
import dflash
from benchmark import compare_unary_preparations, dflash2_execution_configs, run_method
from dflash2_tree import (
    DFLASH2_ORIGINAL_DDTREE,
    DFLASH2_PAIRWISE_K16,
    DFLASH2_UNARY_K16,
    DFLASH2_UNARY_K32,
    DFLASH2_UNARY_K64,
    build_dflash2_verifier_tree,
    candidate_count_for_method,
    dflash2_tree_generate,
    proposal_to_lattice,
)
from model.dflash2 import DFlash2DraftModel


def tiny_config() -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=80,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=128,
        layer_types=["full_attention"],
        dflash_config={"target_layer_ids": [0], "mask_token_id": 79},
        block_size=8,
        conv_group_size=4,
        conv_kernel_size=2,
        selector_top_k=16,
        selector_rank=4,
        sample_from_anchor=False,
    )
    config._attn_implementation = "sdpa"
    return config


@pytest.fixture
def draft() -> DFlash2DraftModel:
    torch.manual_seed(42)
    model = DFlash2DraftModel(tiny_config()).eval()
    with torch.no_grad():
        model.candidate_selector.predecessor_codebook.normal_()
        model.candidate_selector.successor_codebook.normal_()
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tied_logits", [False, True])
def test_lean_proposal_preserves_unary_tensors_without_selector(
    draft, dtype, tied_logits, monkeypatch
) -> None:
    draft = draft.to(dtype)
    hidden = torch.randn(1, 7, 8).to(dtype)
    if tied_logits:
        with torch.no_grad():
            draft.lm_head.weight.zero_()
    shared = draft.propose(hidden, torch.tensor([5]), collect_lattice=True)
    selector = Mock(side_effect=AssertionError("lean unary called the selector"))
    monkeypatch.setattr(draft.candidate_selector, "select_path", selector)
    monkeypatch.setattr(draft.candidate_selector, "pairwise_correction", selector)
    monkeypatch.setattr(draft.candidate_selector.hidden_projection, "forward", selector)

    lean = draft.propose_unary(hidden)

    selector.assert_not_called()
    for field in ("candidate_ids", "unary_scores", "unary_logsumexp", "full_unary_logits"):
        assert torch.equal(getattr(lean, field), getattr(shared, field))
    assert not hasattr(lean, "pairwise_final_scores")
    assert not hasattr(lean, "token_ids")


@pytest.mark.parametrize("budget", [7, 16, 32, 64])
@pytest.mark.parametrize(
    "method",
    [DFLASH2_ORIGINAL_DDTREE, DFLASH2_UNARY_K16, DFLASH2_UNARY_K32, DFLASH2_UNARY_K64],
)
@torch.inference_mode()
def test_lean_and_shared_construct_identical_unary_trees(draft, method, budget) -> None:
    hidden = torch.randn(1, 7, 8)
    lean = draft.propose_unary(hidden)
    shared = draft.propose(hidden, torch.tensor([5]), collect_lattice=True)
    count = candidate_count_for_method(lean, method, budget)
    lean_lattice = proposal_to_lattice(lean, count, require_pairwise=False)
    shared_lattice = proposal_to_lattice(shared, count)
    assert set(lean_lattice) == {
        "candidate_token_ids", "candidate_unary_logits", "unary_logsumexp"
    }
    for field, tensor in lean_lattice.items():
        assert torch.equal(tensor, shared_lattice[field])
    lean_nodes, *lean_compiled = build_dflash2_verifier_tree(lean_lattice, method, budget)
    shared_nodes, *shared_compiled = build_dflash2_verifier_tree(
        shared_lattice, method, budget
    )
    assert lean_nodes == shared_nodes
    for left, right in zip(lean_compiled, shared_compiled):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        else:
            assert left == right


def test_pairwise_rejects_unary_only_proposal(draft) -> None:
    with pytest.raises(ValueError, match="requires a conditional"):
        proposal_to_lattice(draft.propose_unary(torch.randn(1, 7, 8)))


@pytest.mark.parametrize("budget", [7, 16, 32, 64])
def test_generator_lean_unary_preserves_outputs_and_acceptance(
    draft, budget, monkeypatch
) -> None:
    target = Qwen3ForCausalLM(tiny_config()).eval()
    monkeypatch.setattr(dflash2_tree, "cuda_time", lambda: 0.0)
    monkeypatch.setattr(dflash, "cuda_time", lambda: 0.0)
    kwargs = dict(
        model=draft,
        target=target,
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=12,
        stop_token_ids=None,
        tree_budget=budget,
        tree_method=DFLASH2_ORIGINAL_DDTREE,
        collect_tree_data=True,
    )
    shared = dflash2_tree_generate(**kwargs, unary_preparation="shared")
    propose = Mock(side_effect=AssertionError("lean generator invoked conditional proposal"))
    monkeypatch.setattr(draft, "propose", propose)
    lean = dflash2_tree_generate(**kwargs)

    propose.assert_not_called()
    assert shared.proposal_preparation == "shared"
    assert lean.proposal_preparation == "lean"
    assert torch.equal(lean.output_ids, shared.output_ids)
    assert lean.matched_draft_tokens_per_round == shared.matched_draft_tokens_per_round
    assert lean.committed_tokens_per_round == shared.committed_tokens_per_round
    assert lean.decode_rounds == shared.decode_rounds
    assert lean.decode_rounds > 1
    for left, right in zip(lean.round_metrics, shared.round_metrics):
        assert left["tree"] == right["tree"]
        for field, tensor in left["allocation_lattice"].items():
            assert torch.equal(tensor, right["allocation_lattice"][field])


def test_pairwise_generator_still_uses_conditional_proposal(draft, monkeypatch) -> None:
    target = Qwen3ForCausalLM(tiny_config()).eval()
    monkeypatch.setattr(dflash2_tree, "cuda_time", lambda: 0.0)
    monkeypatch.setattr(dflash, "cuda_time", lambda: 0.0)
    conditional = Mock(wraps=draft.propose)
    monkeypatch.setattr(draft, "propose", conditional)
    monkeypatch.setattr(
        draft, "propose_unary", Mock(side_effect=AssertionError("pairwise used unary proposal"))
    )
    result = dflash2_tree_generate(
        model=draft,
        target=target,
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=4,
        stop_token_ids=None,
        tree_budget=16,
        tree_method=DFLASH2_PAIRWISE_K16,
    )
    assert result.proposal_preparation == "conditional"
    assert conditional.call_count == result.decode_rounds > 0


@pytest.mark.parametrize("mode", ["lean", "shared", "both"])
def test_execution_matrix_names_and_dispatch(mode, monkeypatch) -> None:
    configs = dflash2_execution_configs(
        [(DFLASH2_ORIGINAL_DDTREE, 16), (DFLASH2_PAIRWISE_K16, 16)], mode
    )
    assert len(configs) == (3 if mode == "both" else 2)
    generator = Mock(return_value=object())
    monkeypatch.setattr("benchmark.dflash2_tree_generate", generator)
    for key, (method, budget, preparation) in configs.items():
        run_method(
            key,
            draft_model=object(),
            target=object(),
            tokenizer=SimpleNamespace(eos_token_id=79),
            input_ids=torch.tensor([[1]]),
            max_new_tokens=4,
            temperature=0,
            block_size=8,
            method_key_to_tree_budget={key: budget},
            method_key_to_tree_method={key: method},
            method_key_to_unary_preparation={key: preparation},
            collect_allocation_data=False,
        )
        assert generator.call_args.kwargs["tree_method"] == method
        expected = "shared" if mode == "shared" or "_shared_tb" in key else "lean"
        assert generator.call_args.kwargs["unary_preparation"] == expected


@pytest.mark.parametrize("mutation", [None, "output", "matched", "committed", "prompt", "count"])
def test_unary_audit_checks_every_repetition(mutation) -> None:
    def result():
        return SimpleNamespace(
            output_ids=torch.tensor([[1, 2, 3]]),
            matched_draft_tokens_per_round=[1, 0],
            committed_tokens_per_round=[2, 1],
            prompt_hash="same-input",
        )

    lean, shared = result(), result()
    lean.repetitions = [lean, result()]
    shared.repetitions = [shared, result()]
    second = shared.repetitions[1]
    if mutation == "output":
        second.output_ids = torch.tensor([[1, 2, 4]])
    elif mutation == "matched":
        second.matched_draft_tokens_per_round = [0, 1]
    elif mutation == "committed":
        second.committed_tokens_per_round = [1, 2]
    elif mutation == "prompt":
        second.prompt_hash = "different-input"
    elif mutation == "count":
        shared.repetitions.pop()
    response = {
        "dflash2_original_ddtree_tb16": lean,
        "dflash2_original_ddtree_shared_tb16": shared,
    }
    if mutation:
        with pytest.raises((ValueError, RuntimeError)):
            compare_unary_preparations(response)
    else:
        compare_unary_preparations(response)
        assert all(rep.matches_shared_unary for rep in lean.repetitions)
        assert all(rep.acceptance_matches_shared_unary for rep in lean.repetitions)
