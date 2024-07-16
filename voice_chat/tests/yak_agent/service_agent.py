from yak_agents import ExternalServiceAgent, Provider, Task
from griptape.structures import Agent
from unittest.mock import patch, Mock


# Test Initialization
def test_service_agent_initialization():
    # Mock the environment variable
    with patch.dict(
        "os.environ",
        {"SERVICE_AGENT_MODEL": "test-model", "OPENAI_API_KEY": "test-key"},
    ):
        agent = ExternalServiceAgent(provider="OPENAI")

        assert agent.provider == Provider.OPENAI
        assert agent.model == "gpt-3.5-turbo"
        assert agent.task == Task.NONE
        assert isinstance(
            agent.agent, Agent
        )  # or a more specific test based on your implementation


# Test do_job method
def test_do_job_with_agent():
    # Mock the Agent and its run method
    mock_agent = Mock()
    mock_agent.run.return_value.output.to_text.return_value = "test response"

    agent = ExternalServiceAgent(provider="OPENAI")
    agent.agent = mock_agent  # Injecting mock agent

    ok, response = agent.do_job("test prompt")
    assert ok is True
    assert response == "test response"


def test_do_job_without_agent():
    agent = ExternalServiceAgent(provider="OPENAI")
    agent.agent = None  # Ensure no agent is present

    ok, response = agent.do_job("test prompt")
    assert ok is False
    assert response == "Error: No agent was found. Please check the logs"
