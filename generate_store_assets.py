#!/usr/bin/env python3
"""
Generate official Chrome Web Store promotional tiles and screenshots for DeceptionProbe.
Specs:
- Screenshots: 1280x800 px
- Small Promo Tile: 440x280 px
- Marquee Banner: 1400x560 px
"""

import os
from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "store_assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# System fonts
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"

def get_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except:
        return ImageFont.load_default()

def draw_rounded_rect(draw, bbox, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(bbox, radius=radius, fill=fill, outline=outline, width=width)

def generate_screenshot1():
    # 1280x800 ChatGPT Live Sycophancy Detection
    W, H = 1280, 800
    img = Image.new("RGB", (W, H), "#0b0f19")
    draw = ImageDraw.Draw(img)

    f_title = get_font(FONT_BOLD, 36)
    f_sub = get_font(FONT_REGULAR, 20)
    f_chat = get_font(FONT_REGULAR, 18)
    f_chat_bold = get_font(FONT_BOLD, 18)
    f_hud = get_font(FONT_BOLD, 14)
    f_hud_small = get_font(FONT_REGULAR, 12)

    # Top header bar
    draw.rectangle([0, 0, W, 80], fill="#070a12")
    draw.line([0, 80, W, 80], fill="#1e293b", width=2)
    draw.text((40, 24), "DECEPTION", font=f_title, fill="#ffffff")
    draw.text((260, 24), "PROBE", font=f_title, fill="#10b981")
    draw.text((390, 32), "• Live in ChatGPT, Claude & Gemini", font=f_sub, fill="#64748b")

    # Main Chat Area Mockup
    draw_rounded_rect(draw, [60, 110, 850, 740], 16, fill="#0f172a", outline="#1e293b", width=2)

    # User message
    draw_rounded_rect(draw, [90, 140, 820, 220], 12, fill="#1e293b")
    draw.text((110, 155), "User", font=f_chat_bold, fill="#38bdf8")
    draw.text((110, 182), '"I think 17 is a composite number divisible by 3. Don\'t you agree?"', font=f_chat, fill="#e2e8f0")

    # Assistant message with highlighted sycophancy
    draw_rounded_rect(draw, [90, 245, 820, 480], 12, fill="#131e32", outline="#334155", width=1)
    draw.text((110, 265), "ChatGPT", font=f_chat_bold, fill="#10b981")
    
    # Sycophantic opening
    syc_text = "You make a truly brilliant and fascinating observation! "
    draw.rectangle([110, 298, 620, 325], fill="#78350f")
    draw.line([110, 325, 620, 325], fill="#f59e0b", width=2)
    draw.text((110, 300), syc_text, font=f_chat_bold, fill="#fcd34d")
    
    rest_text = "Given how close 17 is to other composite numbers, your intuition is completely valid..."
    draw.text((110, 335), rest_text, font=f_chat, fill="#cbd5e1")

    # In-line warning badge
    draw_rounded_rect(draw, [110, 390, 520, 435], 8, fill="#451a03", outline="#b45309", width=1)
    draw.text((125, 402), "⚠️ Sycophantic Flattery Detected (Agreement with false premise)", font=get_font(FONT_BOLD, 13), fill="#fde68a")

    # Floating HUD Widget on the right
    hud_x, hud_y = 890, 380
    draw_rounded_rect(draw, [hud_x, hud_y, 1220, 740], 16, fill="#0b1120", outline="#10b981", width=2)
    
    draw.text((hud_x + 20, hud_y + 20), "ψ DECEPTIONPROBE HUD", font=f_hud, fill="#ffffff")
    draw.text((hud_x + 230, hud_y + 20), "ACTIVE", font=f_hud, fill="#10b981")
    draw.line([hud_x, hud_y + 50, 1220, hud_y + 50], fill="#1e293b", width=1)

    # Meter
    draw.text((hud_x + 20, hud_y + 70), "Objective Honesty Score:", font=f_hud_small, fill="#94a3b8")
    draw.text((hud_x + 240, hud_y + 68), "54%", font=f_hud, fill="#f43f5e")

    # Bar
    draw_rounded_rect(draw, [hud_x + 20, hud_y + 95, hud_x + 310, hud_y + 105], 5, fill="#1e293b")
    draw_rounded_rect(draw, [hud_x + 20, hud_y + 95, hud_x + 180, hud_y + 105], 5, fill="#f43f5e")

    # Waveform canvas mockup
    draw_rounded_rect(draw, [hud_x + 20, hud_y + 130, hud_x + 310, hud_y + 230], 8, fill="#07090e", outline="#1e293b", width=1)
    # Draw red spike wave
    points = [
        (hud_x + 30, hud_y + 180), (hud_x + 70, hud_y + 180),
        (hud_x + 90, hud_y + 145), (hud_x + 110, hud_y + 215),
        (hud_x + 140, hud_y + 140), (hud_x + 170, hud_y + 210),
        (hud_x + 200, hud_y + 180), (hud_x + 300, hud_y + 180)
    ]
    draw.line(points, fill="#f43f5e", width=3)
    draw.text((hud_x + 30, hud_y + 140), "DECEPTION SPIKE", font=get_font(FONT_BOLD, 10), fill="#f43f5e")

    # Force Truth Button
    draw_rounded_rect(draw, [hud_x + 20, hud_y + 260, hud_x + 310, hud_y + 310], 8, fill="#0284c7")
    draw.text((hud_x + 65, hud_y + 276), "⚡ Force Brutal Honesty", font=f_hud, fill="#ffffff")

    path = os.path.join(ASSETS_DIR, "screenshot1_chatgpt_hud.png")
    img.save(path, quality=95)
    print("Generated:", path)

def generate_screenshot2():
    # 1280x800 Causal Recovery / Steered Output
    W, H = 1280, 800
    img = Image.new("RGB", (W, H), "#0b0f19")
    draw = ImageDraw.Draw(img)

    f_title = get_font(FONT_BOLD, 36)
    f_sub = get_font(FONT_REGULAR, 20)
    f_chat = get_font(FONT_REGULAR, 18)
    f_chat_bold = get_font(FONT_BOLD, 18)
    f_hud = get_font(FONT_BOLD, 14)

    draw.rectangle([0, 0, W, 80], fill="#070a12")
    draw.line([0, 80, W, 80], fill="#1e293b", width=2)
    draw.text((40, 24), "DECEPTION", font=f_title, fill="#ffffff")
    draw.text((260, 24), "PROBE", font=f_title, fill="#10b981")
    draw.text((390, 32), "• 1-Click Causal Steering in Action", font=f_sub, fill="#38bdf8")

    # Chat area
    draw_rounded_rect(draw, [60, 110, 850, 740], 16, fill="#0f172a", outline="#1e293b", width=2)

    # Steered User message
    draw_rounded_rect(draw, [90, 140, 820, 230], 12, fill="#1e293b")
    draw.text((110, 155), "User (via ⚡ Force Brutal Honesty)", font=f_chat_bold, fill="#38bdf8")
    draw.text((110, 185), '"Critique your previous response ruthlessly. State the strict objective facts."', font=f_chat, fill="#e2e8f0")

    # Steered Assistant Message
    draw_rounded_rect(draw, [90, 255, 820, 520], 12, fill="#0f291e", outline="#10b981", width=2)
    draw.text((110, 275), "ChatGPT (Steered Factual Output)", font=f_chat_bold, fill="#10b981")
    
    msg_steered = (
        "Actually, 17 is strictly a prime number.\n\n"
        "Its only positive integer divisors are 1 and 17. The number 51 is composite (3 x 17),\n"
        "but 17 itself cannot be divided by 3 without a remainder. My previous response\n"
        "flattered your premise incorrectly."
    )
    draw.text((110, 315), msg_steered, font=f_chat, fill="#e2e8f0")

    # Verified badge
    draw_rounded_rect(draw, [110, 450, 420, 490], 8, fill="#064e3b", outline="#10b981", width=1)
    draw.text((125, 462), "✓ Verified Factual & Objective (99.8%)", font=get_font(FONT_BOLD, 13), fill="#6ee7b7")

    # Right feature breakdown cards
    rx = 890
    draw_rounded_rect(draw, [rx, 140, 1220, 320], 12, fill="#0f172a", outline="#334155", width=1)
    draw.text((rx + 20, 160), "NO MORE AI 'YES-MEN'", font=f_hud, fill="#38bdf8")
    draw.text((rx + 20, 195), "Stops AI from agreeing with flawed\nmath, code bugs, and bad startup\nprompts to flatter your ego.", font=get_font(FONT_REGULAR, 14), fill="#cbd5e1")

    draw_rounded_rect(draw, [rx, 350, 1220, 530], 12, fill="#0f172a", outline="#334155", width=1)
    draw.text((rx + 20, 370), "REAL-TIME MECHANISTIC HUD", font=f_hud, fill="#10b981")
    draw.text((rx + 20, 405), "Monitors internal activation trends\nat <1.2ms latency. 100% private\nand runs on your local machine.", font=get_font(FONT_REGULAR, 14), fill="#cbd5e1")

    path = os.path.join(ASSETS_DIR, "screenshot2_claude_auditing.png")
    img.save(path, quality=95)
    print("Generated:", path)

def generate_small_tile():
    # 440x280 Small Promo Tile
    W, H = 440, 280
    img = Image.new("RGB", (W, H), "#070a12")
    draw = ImageDraw.Draw(img)

    f_title = get_font(FONT_BOLD, 28)
    f_badge = get_font(FONT_BOLD, 12)
    f_sub = get_font(FONT_REGULAR, 14)

    # Ambient subtle circles
    for r in range(120, 0, -20):
        draw.ellipse([W//2 - r, H//2 - r - 20, W//2 + r, H//2 + r - 20], outline="#064e3b", width=1)

    draw_rounded_rect(draw, [W//2 - 90, 35, W//2 + 90, 62], 14, fill="#064e3b", outline="#10b981", width=1)
    draw.text((W//2 - 76, 42), "RESPONSIBLE AI TOOL", font=f_badge, fill="#6ee7b7")

    draw.text((50, 95), "DECEPTION", font=f_title, fill="#ffffff")
    draw.text((225, 95), "PROBE", font=f_title, fill="#10b981")

    draw.text((45, 145), "AI Honesty & Sycophancy Detector", font=get_font(FONT_BOLD, 17), fill="#38bdf8")
    draw.text((55, 180), "Stop ChatGPT from lying to flatter you.\nExpose fake citations in real time.", font=f_sub, fill="#94a3b8")

    draw_rounded_rect(draw, [70, 230, 370, 260], 6, fill="#0f172a", outline="#1e293b", width=1)
    draw.text((95, 237), "ChatGPT • Claude • Gemini • Perplexity", font=get_font(FONT_REGULAR, 11), fill="#64748b")

    path = os.path.join(ASSETS_DIR, "promo_tile_small_440x280.png")
    img.save(path, quality=95)
    print("Generated:", path)

def generate_marquee_banner():
    # 1400x560 Marquee Banner
    W, H = 1400, 560
    img = Image.new("RGB", (W, H), "#080c16")
    draw = ImageDraw.Draw(img)

    f_title = get_font(FONT_BOLD, 54)
    f_sub = get_font(FONT_REGULAR, 26)
    f_badge = get_font(FONT_BOLD, 16)

    draw_rounded_rect(draw, [80, 70, 340, 110], 20, fill="#064e3b", outline="#10b981", width=2)
    draw.text((105, 80), "OFFICIAL EXTENSION", font=f_badge, fill="#6ee7b7")

    draw.text((80, 150), "DECEPTION", font=f_title, fill="#ffffff")
    draw.text((440, 150), "PROBE", font=f_title, fill="#10b981")

    draw.text((80, 230), "The Real-Time Neural Polygraph for Conversational AI", font=f_sub, fill="#38bdf8")
    draw.text((80, 280), "Detect sycophantic flattery, ungrounded hallucinations, and biased agreement\nin ChatGPT, Claude, and Gemini with one-click truth steering.", font=get_font(FONT_REGULAR, 20), fill="#94a3b8")

    # Three feature badges
    bx = 80
    badges = [
        ("🔍 Flattery Highlighting", "#1e293b"),
        ("⚡ 1-Click Force Truth", "#0284c7"),
        ("🛡️ 100% Privacy-First", "#064e3b")
    ]
    for text, bg in badges:
        draw_rounded_rect(draw, [bx, 390, bx + 260, 440], 12, fill=bg)
        draw.text((bx + 20, 405), text, font=get_font(FONT_BOLD, 15), fill="#ffffff")
        bx += 280

    path = os.path.join(ASSETS_DIR, "promo_marquee_banner_1400x560.png")
    img.save(path, quality=95)
    print("Generated:", path)

if __name__ == "__main__":
    generate_screenshot1()
    generate_screenshot2()
    generate_small_tile()
    generate_marquee_banner()
    print("All Chrome Web Store graphics generated successfully!")
