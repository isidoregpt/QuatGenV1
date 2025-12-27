"""PDF Export for molecules"""

from typing import List


async def export_to_pdf(
    molecules: List,
    output_path: str,
    title: str = "Quat Generator Pro - Export Report",
    include_summary: bool = True,
    include_pareto_plot: bool = True,
    structures_per_page: int = 6,
    page_size: str = "letter",
    include_properties: bool = True,
    include_scores: bool = True
):
    """Export molecules to PDF report"""
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    
    pagesize = letter if page_size == "letter" else A4
    doc = SimpleDocTemplate(output_path, pagesize=pagesize)
    styles = getSampleStyleSheet()
    elements = []
    
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24, spaceAfter=30)
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 20))
    
    if include_summary:
        summary_text = f"<b>Total Molecules:</b> {len(molecules)}<br/>"
        summary_text += f"<b>Pareto Optimal:</b> {sum(1 for m in molecules if m.is_pareto)}<br/>"
        if molecules:
            avg_eff = sum(m.efficacy_score for m in molecules) / len(molecules)
            summary_text += f"<b>Average Efficacy:</b> {avg_eff:.1f}<br/>"
        elements.append(Paragraph(summary_text, styles['Normal']))
        elements.append(Spacer(1, 20))
    
    if include_scores:
        headers = ['ID', 'SMILES', 'Eff%', 'Safe%', 'Env%', 'SA%']
        data = [headers]
        for mol in molecules[:100]:
            smiles_short = mol.smiles[:30] + "..." if len(mol.smiles) > 30 else mol.smiles
            data.append([str(mol.id), smiles_short, f"{mol.efficacy_score:.0f}", f"{mol.safety_score:.0f}",
                        f"{mol.environmental_score:.0f}", f"{mol.sa_score:.0f}"])
        table = Table(data, colWidths=[0.5*inch, 2.5*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.6*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
    
    doc.build(elements)
