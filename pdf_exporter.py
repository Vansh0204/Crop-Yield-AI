from fpdf import FPDF
import time

class AdvisoryPDF(FPDF):
    def header(self):
        # Green Header Banner
        self.set_fill_color(6, 78, 59)
        self.rect(0, 0, 210, 30, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 18)
        self.cell(0, 20, "AgriAI Farm Advisory", ln=True, align='C')
        self.ln(10)

def export_advisory_pdf(state: dict) -> bytes:
    pdf = AdvisoryPDF()
    pdf.add_page()
    
    # Selection Details
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, f"Target Crop: {state.get('crop', 'Unknown')}", ln=True)
    pdf.cell(0, 10, f"Target Region: {state.get('area', 'Global')}", ln=True)
    pdf.cell(0, 5, f"Issued on: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)
    
    # AI Report Body
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Intelligence Report & Recommendations:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    
    # We use multi_cell to handle long text wrapping
    report_text = state.get('advisory_report', "No report generated.")
    # Standardizing characters for PDF compatibility
    report_text = report_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 7, report_text)
    
    return bytes(pdf.output())
