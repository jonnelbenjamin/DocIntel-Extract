from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

file_name = "./data/dummy_invoice.pdf"

c = canvas.Canvas(file_name, pagesize=letter)
width, height = letter

# Title
c.setFont("Helvetica-Bold", 20)
c.drawString(50, height - 50, "INVOICE")

# Vendor info
c.setFont("Helvetica", 12)
c.drawString(50, height - 100, "Vendor: Acme Analytics LLC")
c.drawString(50, height - 120, "Invoice #: INV-2026-0304")
c.drawString(50, height - 140, "Date: 2026-03-04")
c.drawString(50, height - 160, "Bill To: Contoso Federal Programs")
c.drawString(50, height - 180, "Payment Terms: Net 14")
c.drawString(50, height - 200, "Due Date: 2026-03-18")

# Table header
y = height - 250
c.drawString(50, y, "Description")
c.drawString(300, y, "Hours")
c.drawString(360, y, "Rate")
c.drawString(430, y, "Amount")

# Table rows
rows = [
    ("Data pipeline support", "40", "$220", "$8,800.00"),
    ("GenAI extraction POC", "12", "$275", "$3,300.00"),
    ("Storage + compute", "-", "-", "$380.50"),
]

y -= 20
for desc, hrs, rate, amt in rows:
    c.drawString(50, y, desc)
    c.drawString(300, y, hrs)
    c.drawString(360, y, rate)
    c.drawString(430, y, amt)
    y -= 20

# Total
c.setFont("Helvetica-Bold", 12)
c.drawString(350, y - 20, "Total: $12,480.50")

c.save()

print(f"Created {file_name}")