import asyncio
import types

from src.pipelines.manager import PipelineManager


def run_async(gen):
    out = []
    async def drain():
        async for x in gen:
            out.append(x)
    asyncio.get_event_loop().run_until_complete(drain())
    return out


def test_two_class_flag_via_ui():
    pm = PipelineManager()
    gen = pm.dispatch(
        user_request="create 10 pictures of Tzuyu",
        sources={"google": True},
        options={"organize_curated_as_classes": True, "augment": False},
    )
    logs = run_async(gen)
    assert any("Two-class workflow" in str(x) for x in logs)


def test_single_concept_flow():
    pm = PipelineManager()
    gen = pm.dispatch(
        user_request="create 5 pictures of a cat",
        sources={"google": False},
        options={"organize_curated_as_classes": False, "augment": False},
    )
    logs = run_async(gen)
    assert any("Single-concept workflow" in str(x) for x in logs)


