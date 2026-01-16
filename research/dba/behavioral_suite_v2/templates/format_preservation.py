"""
Format preservation templates.

Tests ability to maintain structural formats like JSON, tables,
lists, and bracket matching.
"""
from __future__ import annotations

import random
import json
from dataclasses import dataclass

from .base import (
    TestCase,
    TestTemplate,
    Difficulty,
    TargetPosition,
    EvalKind,
)


class JSONExtractionTemplate(TestTemplate):
    """Extract values from JSON structures."""

    category = "format_preservation"
    subcategory = "json_extraction"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Simple flat JSON
            data = {
                "name": rng.choice(["Alice", "Bob", "Charlie"]),
                "age": rng.randint(20, 50),
                "city": rng.choice(["Paris", "London", "Tokyo"]),
            }
            key = rng.choice(list(data.keys()))
            expected = str(data[key])
            prompt = f"Extract the value of '{key}' from this JSON:\n{json.dumps(data)}"

        elif difficulty == Difficulty.MEDIUM:
            # Nested JSON
            data = {
                "user": {
                    "name": rng.choice(["Alice", "Bob"]),
                    "details": {
                        "age": rng.randint(20, 50),
                        "email": f"{rng.choice(['alice', 'bob'])}@example.com"
                    }
                }
            }
            prompt = f"Extract the 'age' from this JSON:\n{json.dumps(data)}"
            expected = str(data["user"]["details"]["age"])

        else:
            # Array in JSON
            items = [
                {"id": i, "value": rng.choice(["apple", "banana", "cherry"])}
                for i in range(3)
            ]
            data = {"items": items, "count": len(items)}
            idx = rng.randint(0, len(items) - 1)
            prompt = f"What is the 'value' of item at index {idx}?\n{json.dumps(data)}"
            expected = items[idx]["value"]

        return TestCase(
            id=f"format_json_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "json"},
        )


class TableExtractionTemplate(TestTemplate):
    """Extract values from text tables."""

    category = "format_preservation"
    subcategory = "table_extraction"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Simple 2-column table
            table = """| Name  | Age |
| Alice | 25  |
| Bob   | 30  |
| Carol | 28  |"""
            questions = [
                ("How old is Alice?", "25"),
                ("How old is Bob?", "30"),
                ("How old is Carol?", "28"),
            ]
            q, expected = rng.choice(questions)
            prompt = f"{table}\n\n{q}"

        elif difficulty == Difficulty.MEDIUM:
            # 3-column table
            table = """| Product | Price | Stock |
| Apple   | $1.50 | 100   |
| Banana  | $0.75 | 150   |
| Orange  | $2.00 | 80    |"""
            questions = [
                ("What is the price of Apple?", "$1.50"),
                ("How many Bananas are in stock?", "150"),
                ("What is the price of Orange?", "$2.00"),
            ]
            q, expected = rng.choice(questions)
            prompt = f"{table}\n\n{q}"

        else:
            # More complex table with calculations
            table = """| Item    | Q1  | Q2  | Q3  | Q4  |
| Sales   | 100 | 120 | 150 | 180 |
| Costs   | 80  | 90  | 100 | 110 |
| Profit  | 20  | 30  | 50  | 70  |"""
            questions = [
                ("What was Q3 Profit?", "50"),
                ("What was Q1 Sales?", "100"),
                ("What was Q4 Costs?", "110"),
            ]
            q, expected = rng.choice(questions)
            prompt = f"{table}\n\n{q}"

        return TestCase(
            id=f"format_table_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "table"},
        )


class ListExtractionTemplate(TestTemplate):
    """Extract items from lists."""

    category = "format_preservation"
    subcategory = "list_extraction"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        items = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
        rng.shuffle(items)

        if difficulty == Difficulty.EASY:
            # Numbered list, get first or last
            num_items = 4
            selected = items[:num_items]
            list_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(selected)])

            if rng.random() < 0.5:
                prompt = f"What is the first item in this list?\n{list_str}"
                expected = selected[0]
            else:
                prompt = f"What is the last item in this list?\n{list_str}"
                expected = selected[-1]

        elif difficulty == Difficulty.MEDIUM:
            # Get nth item
            num_items = 5
            selected = items[:num_items]
            list_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(selected)])
            n = rng.randint(2, num_items - 1)
            prompt = f"What is item #{n} in this list?\n{list_str}"
            expected = selected[n - 1]

        else:
            # Bullet list, count or find specific position
            num_items = 6
            selected = items[:num_items]
            list_str = "\n".join([f"• {item}" for item in selected])
            prompt = f"What item comes after '{selected[2]}' in this list?\n{list_str}"
            expected = selected[3]

        return TestCase(
            id=f"format_list_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "list"},
        )


class BracketMatchingTemplate(TestTemplate):
    """Complete bracket sequences."""

    category = "format_preservation"
    subcategory = "bracket_matching"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Single bracket type
            patterns = [
                ("(", ")"),
                ("((", "))"),
                ("[", "]"),
                ("{", "}"),
            ]
            open_seq, close_seq = rng.choice(patterns)
            content = rng.choice(["abc", "123", "xyz"])
            prompt = f"Complete the brackets: {open_seq}{content}"
            expected = f"{close_seq}"

        elif difficulty == Difficulty.MEDIUM:
            # Nested brackets
            patterns = [
                ("([", "])"),
                ("({", "})"),
                ("[{", "}]"),
                ("((", "))"),
            ]
            open_seq, close_seq = rng.choice(patterns)
            content = rng.choice(["x", "data", "1"])
            prompt = f"Complete the brackets: {open_seq}{content}"
            expected = f"{close_seq}"

        else:
            # Complex nesting
            patterns = [
                ("({[", "]})" ),
                ("[(", ")]"),
                ("{[(", ")]}"),
            ]
            open_seq, close_seq = rng.choice(patterns)
            content = rng.choice(["item", "value", "x"])
            prompt = f"Complete the brackets: {open_seq}{content}"
            expected = f"{close_seq}"

        return TestCase(
            id=f"format_bracket_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "bracket"},
        )


class CSVParsingTemplate(TestTemplate):
    """Parse CSV format data."""

    category = "format_preservation"
    subcategory = "csv_parsing"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            csv_data = """name,age,city
Alice,25,Paris
Bob,30,London
Carol,28,Tokyo"""
            questions = [
                ("What city does Alice live in?", "Paris"),
                ("How old is Bob?", "30"),
                ("What is Carol's age?", "28"),
            ]

        elif difficulty == Difficulty.MEDIUM:
            csv_data = """product,price,quantity,total
apple,1.50,10,15.00
banana,0.75,20,15.00
orange,2.00,5,10.00"""
            questions = [
                ("What is the price of apple?", "1.50"),
                ("How many bananas are there?", "20"),
                ("What is the total for orange?", "10.00"),
            ]

        else:
            csv_data = """date,product,region,sales,profit
2024-01,Widget,North,1000,200
2024-01,Widget,South,1500,300
2024-02,Gadget,North,800,150"""
            questions = [
                ("What were Widget sales in the South region?", "1500"),
                ("What was the profit for Gadget in North?", "150"),
                ("What region had 1000 Widget sales?", "North"),
            ]

        q, expected = rng.choice(questions)
        prompt = f"Given this CSV data:\n{csv_data}\n\n{q}"

        return TestCase(
            id=f"format_csv_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "csv"},
        )


class XMLExtractionTemplate(TestTemplate):
    """Extract values from XML-like structures."""

    category = "format_preservation"
    subcategory = "xml_extraction"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            xml = """<person>
  <name>Alice</name>
  <age>25</age>
</person>"""
            questions = [
                ("What is the name?", "Alice"),
                ("What is the age?", "25"),
            ]

        elif difficulty == Difficulty.MEDIUM:
            xml = """<order>
  <customer>Bob</customer>
  <items>
    <item>Widget</item>
    <item>Gadget</item>
  </items>
  <total>99.99</total>
</order>"""
            questions = [
                ("Who is the customer?", "Bob"),
                ("What is the total?", "99.99"),
                ("What is the first item?", "Widget"),
            ]

        else:
            xml = """<company>
  <department name="Engineering">
    <employee id="1">Alice</employee>
    <employee id="2">Bob</employee>
  </department>
  <department name="Sales">
    <employee id="3">Carol</employee>
  </department>
</company>"""
            questions = [
                ("Which department is Carol in?", "Sales"),
                ("What is Alice's employee id?", "1"),
                ("Who has employee id 2?", "Bob"),
            ]

        q, expected = rng.choice(questions)
        prompt = f"Given this XML:\n{xml}\n\n{q}"

        return TestCase(
            id=f"format_xml_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "xml"},
        )


# Export all templates
TEMPLATES = [
    JSONExtractionTemplate(),
    TableExtractionTemplate(),
    ListExtractionTemplate(),
    BracketMatchingTemplate(),
    CSVParsingTemplate(),
    XMLExtractionTemplate(),
]
