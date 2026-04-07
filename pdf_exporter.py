from fpdf import FPDF
import time

class AdvisoryPDF(FPDF):
    def header(self):
        self.set_fill_color(6, 78, 59)
        self.rect(0, 0, 210, 30, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 18)
        self.cell(0, 10, "AgriAI Farm Advisory", ln=True, align='C')

def export_advisory_pdf(state: dict) -> bytes:
    pdf = AdvisoryPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"Crop: {state.get('crop', 'Unknown')}", ln=True)
    return pdf.output()
