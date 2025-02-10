from typing import Annotated, Any, Dict, Sequence, TypedDict

import operator
from langchain_core.messages import BaseMessage


import json


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return {**a, **b}

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]  # 消息序列
    data: Annotated[Dict[str, Any], merge_dicts]  # 数据字典
    metadata: Annotated[Dict[str, Any], merge_dicts]  # 元数据字典



def show_agent_reasoning(output, agent_name):
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")  # 打印代理名称
    
    def convert_to_serializable(obj):
        if hasattr(obj, 'to_dict'):  # 处理Pandas Series/DataFrame
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):  # 处理自定义对象
            return obj.__dict__
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)  # 回退到字符串表示
    
    if isinstance(output, (dict, list)):
        # 将输出转换为可序列化的JSON格式
        serializable_output = convert_to_serializable(output)
        print(json.dumps(serializable_output, indent=2))
    else:
        try:
            # 解析字符串为JSON并美化打印
            parsed_output = json.loads(output)
            print(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            # 如果不是有效的JSON，则回退到原始字符串
            print(output)
    
    print("=" * 48)