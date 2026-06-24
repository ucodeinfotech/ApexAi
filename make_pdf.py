"""Convert work report to PDF using fpdf2"""
from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 5, "Historical Stock Data Pipeline - Work Report", align="C")
            self.ln(8)
    
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

pdf = PDF(orientation="P", unit="mm", format="A4")
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# Title page
pdf.add_page()
pdf.ln(60)
pdf.set_font("Helvetica", "B", 24)
pdf.cell(0, 15, "Historical Stock Data Pipeline", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 16)
pdf.cell(0, 10, "Work Report", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(10)
pdf.set_font("Helvetica", "", 12)
pdf.cell(0, 8, "Angel One SmartAPI  |  1-Minute OHLCV  |  175 Stocks", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 8, "October 2016 - June 2026  (~10 years)", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 8, "5 Timeframes: 1min / 5min / 15min / 1hr / 1day", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 8, "Total Data: ~9.4 GB  |  875 CSV Files", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(15)
pdf.set_font("Helvetica", "I", 10)
pdf.cell(0, 8, "Generated: June 19, 2026", align="C", new_x="LMARGIN", new_y="NEXT")

# Read markdown content
with open("work_report.md") as f:
    lines = f.readlines()

# Parse sections
sections = {}
current_section = "header"
section_lines = []

for line in lines:
    if line.startswith("====") and current_section == "header":
        continue
    if line.startswith("PHASE") or line.startswith("FINAL"):
        if current_section != "header":
            sections[current_section] = section_lines
        current_section = line.strip().strip("=").strip()
        section_lines = [line]
    elif line.startswith("END OF REPORT"):
        sections[current_section] = section_lines
    else:
        section_lines.append(line)
if current_section not in sections:
    sections[current_section] = section_lines

# Helper to add text
def add_text(pdf, text, size=9, bold=False, indent=0):
    style = "B" if bold else ""
    pdf.set_font("Courier", style, size)
    # Handle indentation
    if indent:
        x = pdf.get_x()
        pdf.set_x(x + indent)
    # Handle page breaks gracefully
    text = text.rstrip()
    if text:
        pdf.multi_cell(0, 4.2, text, new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.ln(2)

# Content pages
for section_name, slines in sections.items():
    pdf.add_page()
    
    # Section header
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_fill_color(30, 60, 114)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, f"  {section_name}", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)
    
    # Content
    for line in slines:
        stripped = line.rstrip()
        
        # Section sub-header (lines with ==== or ----)
        if stripped.startswith("==") and len(stripped) > 10:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 60, 114)
            pdf.cell(0, 7, stripped.strip("= "), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
            continue
        
        if stripped.startswith("---") and len(stripped) > 10:
            continue
        
        # Lines with **bold** markers (simulate with indent)
        if stripped.startswith("  ") and stripped.strip():
            # Check if it's a sub-header
            if any(stripped.strip().startswith(p) for p in ["Date", "Duration", "Objective", "Actions", "Method", "Approach"]):
                pdf.set_font("Courier", "B", 9)
                pdf.multi_cell(0, 4.2, stripped, new_x="LMARGIN", new_y="NEXT")
                continue
        
        # Regular content
        add_text(pdf, stripped, size=8.5)

# Add a final page with key numbers
pdf.add_page()
pdf.set_font("Helvetica", "B", 14)
pdf.set_fill_color(30, 60, 114)
pdf.set_text_color(255, 255, 255)
pdf.cell(0, 10, "  KEY STATISTICS SUMMARY", fill=True, new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)
pdf.ln(5)

stats = [
    ("Total Stocks", "175 (157 original + 18 added)"),
    ("Total Files", "875 CSV files (175 stocks x 5 timeframes)"),
    ("Total Data Size", "~9.4 GB"),
    ("Date Coverage", "2016-10-03 to 2026-06-19 (~10 years)"),
    ("API Source", "Angel One SmartAPI (SmartConnect)"),
    ("", ""),
    ("TIME FRAMES", ""),
    ("1-Minute (raw)", "~130 million rows across all stocks"),
    ("5-Minute", "Resampled OHLCV"),
    ("15-Minute", "Resampled OHLCV"),
    ("1-Hour", "Resampled OHLCV"),
    ("Daily", "Resampled OHLCV"),
    ("", ""),
    ("INDEX COVERAGE", ""),
    ("Nifty 50", "50/50 (100%)"),
    ("Nifty Next 50", "~46/50 (92%)"),
    ("Nifty Midcap 100", "~85/100 (85%)"),
    ("Sensex", "30/30 (100%)"),
    ("Bank Nifty", "12/12 (100%)"),
    ("Nifty 200 (approx)", "~155/200 (77%)"),
    ("", ""),
    ("DATA QUALITY", ""),
    ("Full history (2016-2026)", "~130 stocks"),
    ("Stocks with minor gaps", "44 stocks (~30-day API holes)"),
    ("Unavailable (Scrip Master)", "ZOMATO - not in Angel One system"),
    ("", ""),
    ("SCRIPTS DEVELOPED", ""),
    ("comprehensive_fetcher.py", "Initial batch download"),
    ("download_one.py", "Resume download with index tracking"),
    ("fix_missing_timeframes.py", "Generate missing 5min/1hr/1day CSVs"),
    ("backfill_gaps.py", "Re-download missing date ranges"),
    ("download_missing.py", "Download 18 additional stocks"),
    ("quick_verify.py", "File existence & size check"),
    ("check_gaps.py", "Data continuity gap detection"),
    ("deep_verify.py", "Date range & gap analysis"),
    ("save_tokens.py / rebuild_tokens.py", "NSE token map builder"),
    ("generate_report.py", "This report"),
]

pdf.set_font("Courier", "B", 9)
for label, value in stats:
    if not label and not value:
        pdf.ln(3)
        continue
    if not value:
        pdf.set_font("Courier", "B", 10)
        pdf.set_text_color(30, 60, 114)
        pdf.cell(0, 5.5, f"  {label}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Courier", "", 9)
    else:
        pdf.set_font("Courier", "B", 9)
        pdf.cell(50, 5, f"  {label}")
        pdf.set_font("Courier", "", 9)
        pdf.cell(0, 5, value, new_x="LMARGIN", new_y="NEXT")

# Save
output_path = "work_report.pdf"
pdf.output(output_path)
print(f"PDF saved to {output_path} ({os.path.getsize(output_path)/1024:.0f} KB)")
