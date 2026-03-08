from io import BytesIO
from random import choice
from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from azure.storage.filedatalake import DataLakeServiceClient
import os
from dotenv import load_dotenv

load_dotenv()

# Azure Data Lake credentials
STORAGE_ACCOUNT_NAME=os.getenv("STORAGE_ACCOUNT_NAME")
STORAGE_ACCOUNT_KEY=os.getenv("STORAGE_ACCOUNT_KEY")
CONTAINER_NAME=os.getenv("CONTAINER_NAME")

# Initialize Faker for random data
fake = Faker()

# Advisory levels and reasons
ADVISORY_LEVELS = ["Level 1: Exercise Normal Precautions", "Level 2: Exercise Increased Caution", 
                   "Level 3: Reconsider Travel", "Level 4: Do Not Travel"]
ADVISORY_REASONS = ["Crime", "Terrorism", "Civil Unrest", "Health Risks", "Natural Disaster"]

# Azure Data Lake Service Client
service_client = DataLakeServiceClient(
    account_url=f"https://{STORAGE_ACCOUNT_NAME}.dfs.core.windows.net",
    credential=STORAGE_ACCOUNT_KEY,
)
file_system_client = service_client.get_file_system_client(file_system=CONTAINER_NAME)

# Generate and upload dummy travel advisories
def generate_and_upload_advisories(num_advisories=50):
    for i in range(1, num_advisories + 1):
        # Generate dummy travel advisory data
        country = fake.country()
        advisory_level = choice(ADVISORY_LEVELS)
        reasons = ", ".join(fake.random_elements(elements=ADVISORY_REASONS, length=2, unique=True))
        notes = fake.paragraph(nb_sentences=3)

        # Create PDF in memory
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter

        # Header
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, height - 50, "Travel Advisory")

        # Country and Advisory Level
        c.setFont("Helvetica", 14)
        c.drawString(50, height - 100, f"Country: {country}")
        c.drawString(50, height - 120, f"Advisory Level: {advisory_level}")

        # Reasons
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 160, "Reasons:")
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 180, reasons)

        # Additional Notes
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 220, "Additional Notes:")
        c.setFont("Helvetica", 12)
        y = height - 240
        for line in notes.split("\n"):
            c.drawString(50, y, line)
            y -= 20

        c.save()

        # Upload PDF to Azure Data Lake
        pdf_buffer.seek(0)
        file_name = f"travel_advisory_{i}.pdf"
        file_client = file_system_client.get_file_client(file_name)
        file_client.upload_data(pdf_buffer, overwrite=True)
        print(f"Uploaded {file_name} to Azure Data Lake")

if __name__ == "__main__":
    generate_and_upload_advisories()