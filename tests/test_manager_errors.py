import pytest
from tests.conftest import drain_async


@pytest.mark.parametrize("bad_request", ["", "   ", None])
def test_invalid_request_is_rejected(pipeline_manager, bad_request):
    gen = pipeline_manager.dispatch(
        user_request=bad_request or "",
        sources={"google": True},
        options={}
    )
    logs = drain_async(gen)
    assert any("Failed to parse" in str(x) or "Could not determine subject" in str(x) for x in logs)


def test_zero_target_gracefully_handles(pipeline_manager):
    gen = pipeline_manager.dispatch(
        user_request="create 0 pictures of a cat",
        sources={"google": True},
        options={}
    )
    logs = drain_async(gen)
    expected_log = "Target count is zero or invalid. Nothing to create."
    assert any(expected_log in str(log_entry) for log_entry in logs)


def test_unfulfillable_concept_handles_no_results(pipeline_manager, mocker):
    mock_crawler = mocker.MagicMock()
    mock_crawler.scrape = mocker.AsyncMock(return_value=None)
    pipeline_manager.crawlers = {"google": mock_crawler}
    gen = pipeline_manager.dispatch(
        user_request="create 3 pictures of qxzzwwv concept",
        sources={"google": True},
        options={}
    )
    logs = drain_async(gen)
    assert any("returned no metadata" in str(x) or "Loop finished" in str(x) for x in logs)


