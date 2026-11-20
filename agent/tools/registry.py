import json
from typing import Dict, List, Any

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
from agent.tools.base import ToolContext
from agent.tools.base import BaseTool
from agent.tools.grounding import (
    QueryParseTool, ReadImageFilesTool, MasksFilterTool, VlmFilterTool,
    VlmScoreTool, ArgmaxImageAndSegIdTool, SegmentTargetInReferenceTool,
    VlmFrameExpansionTool, ExpandFromSecondaryViewTool, VlmFrameExpansionToolV2, ExpandFromSecondaryViewToolV2,
    SegmentAllTargetObjectTool, ReconstructPointCloudTool, CalculateBboxTool, CentroidCompleteTool, CentroidCompleteToolV2
)

class ToolRegistry:
    def __init__(self, context: ToolContext, args: Any):
        self.context = context
        self.args = args
        self.tools: Dict[str, BaseTool] = {}
        self._register_all()
        print(f"Registered tools ({len(self.tools.keys())}):", self.tools.keys())

    def _register_all(self):
        tool_classes = [
            QueryParseTool, ReadImageFilesTool, MasksFilterTool, VlmFilterTool,
            VlmScoreTool, ArgmaxImageAndSegIdTool, SegmentTargetInReferenceTool,
            SegmentAllTargetObjectTool, ReconstructPointCloudTool, CalculateBboxTool
        ]

        if self.args.ste:
            tool_classes.extend([VlmFrameExpansionTool, ExpandFromSecondaryViewTool])

        if self.args.mg_ste:
            tool_classes.extend([VlmFrameExpansionToolV2, ExpandFromSecondaryViewToolV2])

        if self.args.mge:
            tool_classes.append(CentroidCompleteTool)

        if self.args.ka_mge:
            tool_classes.append(CentroidCompleteToolV2)
        
        for cls in tool_classes:
            tool = cls(self.context)
            self.tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool:
        return self.tools.get(name)

    def get_tool_names(self) -> List[str]:
        return list(self.tools.keys())

    def get_tools_description(self) -> str:
        lines = []
        for tool in self.tools.values():
            schema = tool.schema['function']
            lines.append(f"## Tool: {schema['name']}")
            lines.append(f"Description: {schema['description']}")
            lines.append(f"Parameters: {json.dumps(schema['parameters'], indent=2)}")
            if tool.return_description:
                lines.append(f"Returns: {tool.return_description}")
            lines.append("-" * 30)
        return "\n".join(lines)