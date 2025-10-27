"""
Script to build metadata table from all CSV files
Tạo bảng metadata với cấu trúc phân cấp: ĐỀ MỤC → CHỦ ĐỀ CON
"""

import os
import sys
import pandas as pd
import json
from collections import defaultdict
from typing import Dict, List, Set
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


class MetadataBuilder:
    """Build and manage metadata from medical knowledge base"""

    def __init__(self, kb_dir: str = "medical_knowledge_base"):
        self.kb_dir = kb_dir
        self.metadata = defaultdict(lambda: defaultdict(lambda: {
            "count": 0,
            "questions": [],
            "csv_files": set()
        }))

    def _normalize_column_name(self, text: str) -> str:
        """Normalize column names by removing extra spaces"""
        return " ".join(str(text).strip().split())

    def _normalize_text(self, text) -> str:
        """Normalize text content"""
        if text is None or pd.isna(text):
            return ""
        return str(text).strip()

    def extract_metadata_from_csv(self, csv_path: str) -> None:
        """Extract metadata from a single CSV file"""
        csv_name = os.path.basename(csv_path)

        # Try multiple encodings
        df = None
        for encoding in ['utf-8-sig', 'utf-8', 'cp1258', 'latin-1']:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                break
            except Exception:
                continue

        if df is None:
            print(f"[ERROR] Cannot read file: {csv_name}")
            return

        # Normalize column names
        df.columns = [self._normalize_column_name(col) for col in df.columns]

        # Check for required columns
        required_cols = ["ĐỀ MỤC", "CHỦ ĐỀ CON"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"[WARNING] File {csv_name} missing columns: {missing_cols}")
            return

        print(f"[OK] Processing: {csv_name} ({len(df)} rows)")

        # Extract metadata
        for idx, row in df.iterrows():
            de_muc = self._normalize_text(row.get("ĐỀ MỤC", ""))
            chu_de_con = self._normalize_text(row.get("CHỦ ĐỀ CON", ""))
            ma_so = self._normalize_text(row.get("MÃ SỐ", row.get("STT", "")))
            cau_hoi = self._normalize_text(row.get("CÂU HỎI", row.get("Câu hỏi", "")))

            # Skip empty entries
            if not de_muc and not chu_de_con:
                continue

            # Store metadata
            if chu_de_con:
                self.metadata[de_muc][chu_de_con]["count"] += 1
                self.metadata[de_muc][chu_de_con]["csv_files"].add(csv_name)

                if ma_so and cau_hoi:
                    self.metadata[de_muc][chu_de_con]["questions"].append({
                        "ma_so": ma_so,
                        "cau_hoi": cau_hoi[:100] + "..." if len(cau_hoi) > 100 else cau_hoi
                    })
            elif de_muc:
                # Entry có ĐỀ MỤC nhưng không có CHỦ ĐỀ CON
                if "_no_subtopic_" not in self.metadata[de_muc]:
                    self.metadata[de_muc]["_no_subtopic_"] = {
                        "count": 0,
                        "questions": [],
                        "csv_files": set()
                    }
                self.metadata[de_muc]["_no_subtopic_"]["count"] += 1
                self.metadata[de_muc]["_no_subtopic_"]["csv_files"].add(csv_name)

    def build_from_directory(self) -> None:
        """Build metadata from all CSV files in directory"""
        if not os.path.isdir(self.kb_dir):
            raise FileNotFoundError(f"Knowledge base directory not found: {self.kb_dir}")

        csv_files = [f for f in os.listdir(self.kb_dir) if f.lower().endswith('.csv')]

        if not csv_files:
            print("[ERROR] No CSV files found!")
            return

        print(f"\nFound {len(csv_files)} CSV files")
        print("=" * 60)

        for csv_file in csv_files:
            csv_path = os.path.join(self.kb_dir, csv_file)
            self.extract_metadata_from_csv(csv_path)

    def get_metadata_summary(self) -> Dict:
        """Get summary statistics of metadata"""
        summary = {
            "total_topics": len(self.metadata),
            "total_subtopics": sum(len(subtopics) for subtopics in self.metadata.values()),
            "total_questions": sum(
                sum(info["count"] for info in subtopics.values())
                for subtopics in self.metadata.values()
            ),
            "topics": []
        }

        for de_muc, subtopics in sorted(self.metadata.items()):
            topic_info = {
                "de_muc": de_muc,
                "total_questions": sum(info["count"] for info in subtopics.values()),
                "subtopics": []
            }

            for chu_de_con, info in sorted(subtopics.items()):
                if chu_de_con == "_no_subtopic_":
                    continue

                topic_info["subtopics"].append({
                    "chu_de_con": chu_de_con,
                    "count": info["count"],
                    "csv_files": sorted(list(info["csv_files"]))
                })

            summary["topics"].append(topic_info)

        return summary

    def export_to_csv(self, output_path: str = "metadata_table.csv") -> None:
        """Export metadata to CSV file"""
        rows = []

        for de_muc, subtopics in sorted(self.metadata.items()):
            for chu_de_con, info in sorted(subtopics.items()):
                if chu_de_con == "_no_subtopic_":
                    continue

                rows.append({
                    "ĐỀ MỤC": de_muc,
                    "CHỦ ĐỀ CON": chu_de_con,
                    "Số lượng câu hỏi": info["count"],
                    "Tệp nguồn": ", ".join(sorted(list(info["csv_files"])))
                })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n[OK] Exported metadata table to: {output_path}")
        return df

    def export_to_json(self, output_path: str = "metadata_structure.json") -> None:
        """Export metadata to JSON file with full structure"""
        # Convert sets to lists for JSON serialization
        json_data = {}
        for de_muc, subtopics in self.metadata.items():
            json_data[de_muc] = {}
            for chu_de_con, info in subtopics.items():
                if chu_de_con == "_no_subtopic_":
                    continue
                json_data[de_muc][chu_de_con] = {
                    "count": info["count"],
                    "csv_files": sorted(list(info["csv_files"])),
                    "sample_questions": info["questions"][:5]  # First 5 questions
                }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"[OK] Exported JSON structure to: {output_path}")

    def print_tree(self) -> None:
        """Print metadata as a tree structure"""
        print("\n" + "=" * 60)
        print("METADATA STRUCTURE TREE")
        print("=" * 60)

        for de_muc, subtopics in sorted(self.metadata.items()):
            total = sum(info["count"] for info in subtopics.values())
            print(f"\n[TOPIC] {de_muc} ({total} cau hoi)")

            for chu_de_con, info in sorted(subtopics.items()):
                if chu_de_con == "_no_subtopic_":
                    continue

                files = ", ".join(sorted(list(info["csv_files"])))
                print(f"   +-- {chu_de_con} ({info['count']} cau hoi)")
                print(f"   |   +-- Nguon: {files}")


def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("BUILDING METADATA TABLE FROM MEDICAL KNOWLEDGE BASE")
    print("=" * 60)

    # Initialize builder
    builder = MetadataBuilder(kb_dir="medical_knowledge_base")

    # Build metadata from all CSV files
    builder.build_from_directory()

    # Print tree structure
    builder.print_tree()

    # Get and print summary
    summary = builder.get_metadata_summary()
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total ĐỀ MỤC: {summary['total_topics']}")
    print(f"Total CHỦ ĐỀ CON: {summary['total_subtopics']}")
    print(f"Total Questions: {summary['total_questions']}")

    # Export to CSV
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)

    df = builder.export_to_csv("medical_knowledge_base/metadata_table.csv")
    print(f"\nPreview of metadata table:")
    print(df.head(10).to_string(index=False))

    # Export to JSON
    builder.export_to_json("medical_knowledge_base/metadata_structure.json")

    print("\n" + "=" * 60)
    print("METADATA BUILD COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
