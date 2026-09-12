"""FAQ pairs already written in a text file, extracted with no model.

python examples/build_faq.py
"""

from __future__ import annotations

from pathlib import Path

from wyra import build_dataset
from wyra.generators import QAPairExtractor

HERE = Path(__file__).parent

result = build_dataset(
    HERE / "data" / "faq_pt.txt",
    HERE / "out" / "faq",
    generator=QAPairExtractor(system="Responda de forma direta e em português."),
)

print(result.report.summary())
for example in result.train:
    print(f"- {example.messages[1].content}")
