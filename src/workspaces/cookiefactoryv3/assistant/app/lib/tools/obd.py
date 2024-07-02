from typing import Any, Dict, List, Optional

from langchain import LLMChain, PromptTemplate
from langchain.agents import tool
from langchain.chains.base import Chain
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from ..llm import get_bedrock_text, get_processed_prompt_template

class DummyData:
    @staticmethod
    def get_equipment_data():
        return {
            "Labeling Belt(라벨링 벨트)": {
                "Status": "Starved",
                "Good Products": 0,
                "Rejected Products": 0
            },
            "Box Sealer(박스 씰러)": {
                "Status": "Starved",
                "Good Products": 0,
                "Rejected Products": 0
            },
            "Conveyor Left Turn(왼쪽 컨베이어)": {
                "Buffered Products": 0
            },
            "Conveyor Left Turn(오른쪽 컨베이어)": {
                "Buffered Products": 4
            },
            "Conveyor Vertical(수직 컨베이어)": {
                "Status": "Starved",
                "Good Products": 0,
                "Rejected Products": 0
            },
            "Box Erector(박스 이렉터)": {
                "Status": "Blocked",
                "Good Products": 1,
                "Rejected Products": 0
            },
            "Plastic Liner(플라스틱 라이너)": {
                "Status": "Blocked",
                "Good Products": 0,
                "Rejected Products": 0
            },
            "Cookie Inspector(쿠키 인스펙터)": {
                "Status": "Down",
                "Good Products": 0,
                "Rejected Products": 20
            },
            "Freezer Tunnel(프리저 터널)": {
                "Status": "Blocked",
                "Temperature": -25.63,
                "Speed (RPM)": 0
            },
            "Cookie Former(쿠키 포머)": {
                "Status": "Blocked",
                "Humidity": 52.12,
                "Temperature": 9.22
            }
        }

llm = get_bedrock_text()

def get_equipment_info():
    data = DummyData.get_equipment_data()
    info = ""
    for equipment, values in data.items():
        info += f"{equipment}:\n"
        for key, value in values.items():
            if key in ["Good Products", "Rejected Products", "Buffered Products"]:
                info += f"  {key}: {value} per minute\n"
            elif key == "Moisture":
                info += f"  {key}: {value}%\n"
            elif key == "Temperature":
                info += f"  {key}: {value}°C\n"
            else:
                info += f"  {key}: {value}\n"
        info += "\n"
    return info.strip()

prompt_template = get_processed_prompt_template("""
You are an AI assistant specializing in factory operations. You have access to the following equipment data:

{equipment_info}

Please answer the following question about factory equipment:

{question}

Provide a detailed and accurate response based on the given information. Consider all aspects of the equipment data, including status, production rates(생산량), environmental conditions, and operational parameters.
All answers must be in Korean.
""")

def get_tool_metadata():
    return {
        "name": "factory_equipment",
        "description": "Useful for answering questions about factory equipment, including status, production rates(생산량)), \
            environmental conditions, and operational parameters. \
            Input to this tool is a question about factory equipment. \
            Output is the answer based on the available equipment data.",
    }
    
@tool
def run(input: str) -> str:
    """Answer questions about factory equipment using the provided data."""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["equipment_info", "question"],
    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt
    )

    output = chain.run({"equipment_info": get_equipment_info(), "question": input})

    return output

class OBDChain(Chain):
    """Chain that answers questions about factory equipment."""

    @property
    def input_keys(self) -> List[str]:
        return ['question']

    @property
    def output_keys(self) -> List[str]:
        return ['text']

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["equipment_info", "question"],
        )

        chain = LLMChain(
            llm=llm,
            prompt=prompt
        )

        output = chain.run({"equipment_info": get_equipment_info(), "question": inputs['question']})

        return {
            'text': output
        }

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["equipment_info", "question"],
        )

        chain = LLMChain(
            llm=llm,
            prompt=prompt
        )

        output = await chain.arun({"equipment_info": get_equipment_info(), "question": inputs['question']})

        return {
            'text': output
        }
