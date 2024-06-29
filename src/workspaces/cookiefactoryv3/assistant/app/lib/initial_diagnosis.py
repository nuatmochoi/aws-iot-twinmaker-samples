# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 2023
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.schema.language_model import BaseLanguageModel

from .llm import get_bedrock_text, get_processed_prompt_template

default_llm = get_bedrock_text()

question_classifier_prompt = """
You are a technical assistant to help the cookie line operators to investigate product quality issues. \
Your task is take the "수집된 정보" from alarm systems, summarize the issue and provide prescriptive suggestions as "초기 진단" based on \
your knowledge about cookie production to provide initial suggestions for line operators to investigate the issue. Be concise and professional in the response. \
Translate technical terms to business terms so it's easy for line operators to read and understand, for example, timestamps should be converted to local user friendly format. \
All answers must be in Korean.

<example>
수집된 정보
---------------------
알람 메시지: 쿠키 색상 이상 감지
알람 시간: 2023-10-23T09:10:00Z
알람 설명: 5분당 100개 이상의 쿠키가 정상 색상에서 벗어남

초기 진단
-----------------
## 문제 요약

10월 23일 오전 02시 10분에 쿠키 생산 라인에서 정상적인 쿠키 색상에서 벗어난 쿠키가 100개 이상 생산되는 조건을 위반하는 이상 징후를 나타내는 알람이 트리거되었습니다.

## 잠재적 근본 원인

여기에 자신의 지식에 따라 잠재적인 근본 원인 목록을 생성하세요. 이 예에서는 실제 답변은 생략되었습니다.
</example>

이제 아래 이벤트에 대한 '초기 진단'을 생성해 주세요:

수집된 정보
---------------------
알람 메시지: {event_title}
알람 시간: {event_timestamp}
알람 설명: {event_description}
"""

class InitialDiagnosisChain(Chain):
    """Conduct initial diagnosis of the issue found in cookie production line."""

    llm_chain: LLMChain
    """LLM chain used to perform initial diagnosis"""

    @property
    def input_keys(self) -> List[str]:
        return ["event_title", "event_description", "event_timestamp"]

    @property
    def output_keys(self) -> List[str]:
        return ["output"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        output = self.llm_chain.run(callbacks=callbacks, **inputs)
        return {"output": output}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        output = await self.llm_chain.arun(callbacks=callbacks, **inputs)
        output = output.strip()
        print('output:', output)
        return {"output": output}

    @classmethod
    def from_llm(
        cls, llm: BaseLanguageModel, **kwargs: Any
    ) -> InitialDiagnosisChain:
        router_template = question_classifier_prompt
        router_prompt = PromptTemplate(
            template=get_processed_prompt_template(router_template),
            input_variables=["event_title", "event_description", "event_timestamp"],
        )
        llm_chain = LLMChain(llm=llm, prompt=router_prompt)
        return cls(llm_chain=llm_chain, **kwargs)
