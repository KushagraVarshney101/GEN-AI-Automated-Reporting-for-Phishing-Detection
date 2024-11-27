import os
import re
import requests
import google.generativeai as GENAI
from PIL import Image as PILImage  # Rename PIL Image to PILImage
import chardet
import warnings
import sys
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageTemplate, Frame
from reportlab.lib.units import inch
from transformers import BlipProcessor, BlipForConditionalGeneration  # For image captioning
from reportlab.platypus.flowables import PageBreak

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_VERBOSITY'] = 'ERROR'

load_dotenv()

GENAI.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = GENAI.GenerativeModel("gemini-1.5-flash")

# Load the BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def extract_base_domain(url):
    pattern = r'^https?:\/\/(?:www\.)?([^\/\?\:]+)'
    match = re.match(pattern, url)
    if match:
        base_domain = match.group(1)
        parts = base_domain.split('.')
        if len(parts) > 1:
            return '.'.join(parts[:-1])
        return base_domain
    return None

def capture_screenshot(api_key, url, output_file):
    api_url = f"https://api.screenshotmachine.com?key={api_key}&url={url}&dimension=1920x1080"
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        with open(output_file, 'wb') as file:
            file.write(response.content)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to capture screenshot: {e}")

def generate_image_caption(image_path):
    try:
        # Open the image using PIL
        image = PILImage.open(image_path).convert("RGB")
        
        # Process the image and generate the caption
        inputs = processor(images=image, return_tensors="pt")
        out = caption_model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        raise RuntimeError(f"Error generating image caption: {e}")

def generate_response(prompt):
    try:
        response = model.generate_content(prompt)
        if hasattr(response, '_done') and response._done:
            if hasattr(response, '_result') and response._result:
                result = response._result
                if hasattr(result, 'candidates') and result.candidates:
                    candidate = result.candidates[0]
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        response_text = "\n".join(part.text for part in candidate.content.parts)
                        return response_text
        return "No response available."
    except Exception as e:
        raise RuntimeError(f"Error generating response: {e}")

def clean_response(response):
    lines = response.split('\n')
    cleaned_lines = []
    for line in lines:
        cleaned_line = line.replace('*', '').replace('#', '').strip()
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    return '\n'.join(cleaned_lines)
def add_footer(canvas, doc):
    footer_text = "Confidential: Authorized recipients only; unauthorized use or distribution is prohibited."
    canvas.saveState()
    canvas.setFont('Helvetica-Bold', 10)
    footer_y = 30
    margin = doc.leftMargin
    page_width = doc.width
    canvas.setStrokeColorRGB(0.6, 0.6, 0.6)
    canvas.setLineWidth(0.8)
    canvas.line(margin, footer_y + 30, margin + page_width, footer_y + 30)
    text_width = canvas.stringWidth(footer_text, 'Helvetica-Bold', 10)
    text_x = margin + (page_width - text_width) / 2  # Center the text
    canvas.drawString(text_x, footer_y, footer_text)
    canvas.restoreState()

def format_response(response, result):
    if result == "Phishing":
        replacements = {
            "Content:": "\nContent:\n",
            "Phishing Characteristics:": "\nPhishing Characteristics:\n",
            "Possible Red Flags:": "\nPossible Red Flags:\n",
            "Recommendations:": "\nRecommendations:\n",
            "Conclusion:": "\nConclusion:\n",
        }
    elif result == "Legitimate":
        replacements = {
            "Content:": "\nContent:\n",
            "Legitimate Characteristics:": "\nLegitimate Characteristics:\n",
            "Possible Green Flags:": "\nPossible Green Flags:\n",
            "Recommendations:": "\nRecommendations:\n",
            "Impacts:": "\nImpacts:\n",
        }
    else:
        raise ValueError("Invalid result argument. Must be 'Phishing' or 'Legitimate'.")

    for key, value in replacements.items():
        response = response.replace(key, value)

    return response
def save_result_to_txt(result, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(result)
    except Exception as e:
        raise RuntimeError(f"Error saving text file: {e}")

def save_result_to_pdf(result, file_path, image_path):
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(name='Title', fontSize=20, fontName='Helvetica-Bold', spaceAfter=6)
    bold_underline_style = ParagraphStyle(name='BoldUnderline', fontSize=14, fontName='Helvetica-Bold', spaceAfter=6, underline=True)
    bold_style = ParagraphStyle(name='Bold', fontSize=12, fontName='Helvetica-Bold', spaceAfter=6)
    body_style = ParagraphStyle(name='Body', fontSize=12, spaceAfter=6, leading=14)

    # Process the text to find the title
    for line in result.split('\n'):
        line = line.strip()
        if line:
            if line in ["Phishing Analysis Report", "Legitimate Analysis Report"]:
                story.append(Paragraph(line, title_style))
                story.append(Spacer(1, 12))

                # Add the image right after the title
                if image_path:
                    img = Image(image_path, width=4*inch, height=3*inch)
                    story.append(img)
                    story.append(Spacer(1, 12))

            elif line in ["Analysis:", "Content:", "Phishing Characteristics:", "Possible Red Flags:", "Recommendations:", "Conclusion:", "Legitimate Characteristics:", "Possible Green Flags:", "Impacts:"]:
                story.append(Paragraph(line, bold_underline_style))
            elif line in ["1. Lack of Context:", "2. Suspicious Request:", "3. Lack of Security Indicators:", "Overall:", "Important Note:"]:
                story.append(Paragraph(line, bold_style))
            else:
                story.append(Paragraph(line, body_style))
            story.append(Spacer(1, 6))

    frame = Frame(doc.leftMargin, doc.bottomMargin + 40, doc.width, doc.height - 70, id='normal')
    template = PageTemplate(id='FooterTemplate', frames=[frame], onPage=add_footer)
    doc.addPageTemplates([template])

    try:
        doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
        print(file_path)
    except Exception as e:
        raise RuntimeError(f"Error saving PDF file: {e}")

def main(url, result):
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    
    base_domain = extract_base_domain(url)
    if not base_domain:
        raise ValueError(f"Invalid URL provided: {url}")
    
    text_file_path = f"uploads/TextFiles/{base_domain}.txt"
    pdf_file_path = f"uploads/pdfs/{base_domain}.pdf"
    screenshot_file_path = f"uploads/screenshots/{base_domain}.png"
    image_path = f"uploads/screenshots/{base_domain}.png"

    api_key = os.getenv("SCREENSHOT_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the SCREENSHOT_API_KEY environment variable.")
    
    try:
        capture_screenshot(api_key, url, screenshot_file_path)
        caption = generate_image_caption(screenshot_file_path)  # Get image caption
        result_from_image = caption  # Use the caption as the text for analysis
        
        save_result_to_txt(result_from_image, text_file_path)

        prompt = (
            f"{'Phishing Analysis Report' if result == 'Phishing' else 'Legitimate Analysis Report'}\n\n"
            f"Analyze the following content for {'phishing' if result == 'Phishing' else 'legitimate'} characteristics and provide a detailed analysis.\n\n"
            f"Provide only content, {'Analysis, Phishing Characteristics, Possible Red Flags, Recommendations, Conclusion' if result == 'Phishing' else 'Legitimate Characteristics, Possible Green Flags, Impacts'}.\n\n"
            f"{'Content should not exceed three line. Phishing Characteristics with 4 points only. Possible Red Flags with 3 points only. Recommendations with 3 points only. Conclusion' if result == 'Phishing' else 'Content should not exceed one line. Legitimate Characteristics with 4 points only. Possible Green Flags with 3 points only. Impacts with 3 points only'}.\n\n"
            f"Provide the content in a way that it can be incorporated into a PDF format easily with structure containing Analysis, Content.\n\n"
            f"Dont provide this heading Analysis: in response provide {'Phishing Analysis Report' if result == 'Phishing' else 'Legitimate Analysis Report'}\n\n"
            f"Content:\n{result_from_image}"
        )

        response = generate_response(prompt)
        cleaned_response = clean_response(response)
        formatted_response = format_response(cleaned_response, result)
        save_result_to_pdf(formatted_response, pdf_file_path, image_path)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Invalid number of arguments. Expected URL and result.")

    url_arg = sys.argv[1]
    result_arg = sys.argv[2]

    # Check if the second argument is either 'Phishing' or 'Legitimate'
    if result_arg not in ["Phishing", "Legitimate"]:
        raise ValueError("Invalid result argument. Must be 'Phishing' or 'Legitimate'.")

    main(url_arg, result_arg)

