"""Multi-agent procurement audit.

    purchase request
          |
          +--> Risk & Compliance agent --+
          +--> Tax & Treasury agent    --+--> CFO synthesis (one LLM call) --> audit memo
          +--> Financial Control agent --+

Each specialist is a LangChain v1 agent (create_agent) with its own prompt, tools and
a shared middleware stack (agent/middleware.py: PII redaction, call limits, retries,
tool-call logging). The ProcurementSupervisor runs them one after another, always in
the same order, and hands their reports to the CFO prompt.
Note: create_agent is built on LangGraph, but we never write graph code.
"""

from langchain.agents import create_agent
from logger import GLOBAL_LOGGER as log
from rag import extract_text, get_llm

from .middleware import build_agent_middleware
from .prompts import (
    CONTROL_AGENT_PROMPT,
    RISK_AGENT_PROMPT,
    SYNTHESIS_PROMPT_TEMPLATE,
    TAX_AGENT_PROMPT,
)
from .tools import (
    calculate_cross_border_tax,
    categorize_expense,
    check_sanctions_list,
    get_vendor_credit_score,
    validate_fx_hedge,
)


def create_specialized_agent(tools, system_prompt):
    """An agent = the LLM + a set of tools + a system prompt (+ middleware around the loop)."""
    return create_agent(
        model=get_llm(),
        tools=tools,
        system_prompt=system_prompt,
        middleware=build_agent_middleware(),
    )


class ProcurementSupervisor:
    """Run the three specialist agents in sequence and synthesize their reports into a memo."""

    def __init__(self):
        self.llm = get_llm()
        self.risk_agent = create_specialized_agent(
            [check_sanctions_list, get_vendor_credit_score], RISK_AGENT_PROMPT
        )
        self.tax_agent = create_specialized_agent(
            [calculate_cross_border_tax, validate_fx_hedge],
            system_prompt=TAX_AGENT_PROMPT,
        )
        self.control_agent = create_specialized_agent(
            tools=[categorize_expense],
            system_prompt=CONTROL_AGENT_PROMPT,
        )

    @staticmethod
    def _invoke_agent(agent, request: str) -> str:
        """Run one agent and return the text of its final message."""
        result = agent.invoke({"messages": [{"role": "user", "content": request}]})
        return result["messages"][-1].text

    def run_audit(self, request: str) -> dict:
        log.info("Starting audit process...")
        log.info("Running Risk & Compliance agent...")
        risk_result = self._invoke_agent(self.risk_agent, request)
        log.info("Risk & Compliance agent completed. result: %s", risk_result)

        log.info("Running Tax & Treasury agent...")
        tax_result = self._invoke_agent(self.tax_agent, request)
        log.info("Tax & Treasury agent completed. result: %s", tax_result)

        log.info("Running Financial Control agent...")
        control_result = self._invoke_agent(self.control_agent, request)
        log.info("Financial Control agent completed. result: %s", control_result)

        synthesis_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            risk_result=risk_result,
            tax_result=tax_result,
            control_result=control_result,
        )

        log.info("Starting audit phase: 4- CFO synthesis...")
        cfo_response = self.llm.invoke(synthesis_prompt)
        cfo_memo = extract_text(cfo_response.content)
        log.info("CFO synthesis completed. result: %s", cfo_memo)

        return {
            "risk_result": risk_result,
            "tax_result": tax_result,
            "control_result": control_result,
            "cfo_memo": cfo_memo,
        }
