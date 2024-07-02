# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 2023
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
import logging

logging.getLogger('botocore').setLevel(logging.DEBUG)

import langchain
langchain.debug = True
langchain.verbose = True

import chainlit as cl
from chainlit.context import context

from lib.router import LLMRouterChain, MultiRouteChain, create_routes
from lib.llm import get_bedrock_text
from lib.context_memory import EntityContextMemory
from lib.tools.qa import init_db
from lib.initial_diagnosis import InitialDiagnosisChain

print('initializing vector db')

init_db(os.path.join(os.path.dirname(__file__), '..', 'public', './freezer_tunnel_manual.pdf'))
init_db(os.path.join(os.path.dirname(__file__), '..', 'public', './biscuit_cookie_manual.pdf'))

print('done initializing vector db')


welcome_message="""
안녕하세요. GS네오텍 AI 어시스턴트 "WiseN AI"입니다. 제과 공장의 생산 라인과 관련된 문제 진단과 해결을 도와드릴 수 있습니다.
"""

welcome_message_with_event=welcome_message + """

이벤트 [#{event_id}](https://example.com/issue/{event_id})가 발생했습니다. 문제에 대한 초기 진단을 실행하고 싶으신가요?
"""


@cl.on_chat_start
async def start():
  LLMRouterChain.update_forward_refs()
  MultiRouteChain.update_forward_refs()
  
  memory = EntityContextMemory()
  routes = create_routes(memory)
  llm_chain = MultiRouteChain.from_prompts(llm=get_bedrock_text(), prompt_infos=routes)

  cl.user_session.set("chain", llm_chain)

  print('user data',  context.session.user_data)
  event_id = context.session.user_data.get('event_id')
  
  if event_id:
    actions = [
        cl.Action(name="initial_chat_actions", value="initial_diagnosis", label="이슈 진단 실행", description="Run Issue Diagnosis"),
    ]
    message = welcome_message_with_event.format(event_id=event_id)
    await cl.Message(content=message, actions=actions).send()
  else:
    message = welcome_message
    await cl.Message(content=message).send()
  

@cl.on_message
async def main(message, context):
  llm_chain = cl.user_session.get("chain")

  res = await llm_chain.acall(
    message,
    callbacks=[cl.AsyncLangchainCallbackHandler()])
  
  await cl.Message(content=res["text"]).send()


@cl.action_callback("initial_chat_actions")
async def on_action(action):
  event_title = context.session.user_data.get('event_title')
  event_description = context.session.user_data.get('event_description')
  event_timestamp = context.session.user_data.get('event_timestamp')
  
  await cl.Message(content=f"이벤트 초기 진단 실행 중...").send()
  await action.remove()
  
  cb = cl.AsyncLangchainCallbackHandler()
  chain = InitialDiagnosisChain.from_llm(llm=get_bedrock_text())
  
  res = await chain.acall({
    'event_title': event_title,
    'event_description': event_description,
    'event_timestamp': event_timestamp
  }, callbacks=[cb])

  actions = [
      cl.Action(name="agent_actions", value="3d", label="Show in 3D", description="Show in 3D")
  ]
  
  await cl.Message(content=res['output'], actions=actions).send()

@cl.action_callback("agent_actions")
async def on_action(action):
  event_entity_id = context.session.user_data.get('event_entity_id')
  if action.value == "3d":
    await cl.Message(content=f"문제 현장으로 이동합니다.").send()
    await action.remove()
    # point camera to the entity id
    await context.session.emit('view', event_entity_id)
