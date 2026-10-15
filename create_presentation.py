"""
Generate professional 7-slide PowerPoint presentation for
Hospital Bed Occupancy Forecasting System (CarePulse Analytics).
Theme: Option B - Clean Light Corporate Theme.
Widescreen 16:9 layout with structured cards and modern typography.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# Initialize Presentation
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_slide_layout = prs.slide_layouts[6]  # Blank slide

# Color Palette (Option B: Clean Light Corporate)
COLOR_BG = RGBColor(255, 255, 255)           # Pure White
COLOR_NAVY = RGBColor(15, 23, 42)            # #0F172A Deep Navy
COLOR_TITLE = RGBColor(30, 41, 59)           # #1E293B Dark Slate
COLOR_BODY = RGBColor(51, 65, 85)            # #334155 Slate
COLOR_MUTED = RGBColor(100, 116, 139)        # #64748B Muted Slate
COLOR_PURPLE = RGBColor(124, 58, 237)        # #7C3AED Primary Violet
COLOR_INDIGO = RGBColor(79, 70, 229)         # #4F46E5 Accent Indigo
COLOR_PILL_BG = RGBColor(243, 232, 255)      # #F3E8FF Light Violet
COLOR_PILL_TXT = RGBColor(109, 40, 217)      # #6D28D9 Dark Violet
COLOR_CARD_BG = RGBColor(248, 250, 252)      # #F8FAFC Light Slate Box
COLOR_CARD_BORDER = RGBColor(226, 232, 240)  # #E2E8F0 Subtle Border
COLOR_EMERALD = RGBColor(16, 185, 129)       # #10B981 Success / Health
COLOR_AMBER = RGBColor(245, 158, 11)         # #F59E0B Warning / Alert
COLOR_CRIMSON = RGBColor(239, 68, 68)        # #EF4444 Critical Alert
COLOR_WHITE = RGBColor(255, 255, 255)


def add_header(slide, badge_text, title_text, subtitle_text, slide_num=None):
    """Adds a standard modern header and footer to a slide."""
    # Top badge pill
    badge_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.8), Inches(0.45), Inches(3.2), Inches(0.35)
    )
    badge_box.fill.solid()
    badge_box.fill.fore_color.rgb = COLOR_PILL_BG
    badge_box.line.color.rgb = COLOR_PURPLE
    badge_box.line.width = Pt(1)
    
    tf_badge = badge_box.text_frame
    tf_badge.word_wrap = True
    tf_badge.margin_left = Inches(0.1)
    tf_badge.margin_right = Inches(0.1)
    tf_badge.margin_top = Inches(0.04)
    tf_badge.margin_bottom = Inches(0.04)
    p_b = tf_badge.paragraphs[0]
    p_b.text = badge_text.upper()
    p_b.font.size = Pt(10)
    p_b.font.bold = True
    p_b.font.color.rgb = COLOR_PILL_TXT
    p_b.alignment = PP_ALIGN.CENTER

    # Main Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.85), Inches(11.7), Inches(0.6))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
    p_t = tf_title.paragraphs[0]
    p_t.text = title_text
    p_t.font.size = Pt(22)
    p_t.font.bold = True
    p_t.font.color.rgb = COLOR_NAVY

    # Subtitle
    sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.45), Inches(11.7), Inches(0.45))
    tf_sub = sub_box.text_frame
    tf_sub.word_wrap = True
    tf_sub.margin_left = tf_sub.margin_top = tf_sub.margin_right = tf_sub.margin_bottom = 0
    p_s = tf_sub.paragraphs[0]
    p_s.text = subtitle_text
    p_s.font.size = Pt(13)
    p_s.font.color.rgb = COLOR_MUTED

    # Bottom footer line & text
    if slide_num is not None:
        foot_sep = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(7.08), Inches(11.7), Inches(0.015))
        foot_sep.fill.solid()
        foot_sep.fill.fore_color.rgb = COLOR_CARD_BORDER
        foot_sep.line.fill.background()

        foot_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.13), Inches(8.0), Inches(0.3))
        tf_f = foot_box.text_frame
        tf_f.margin_left = tf_f.margin_top = tf_f.margin_right = tf_f.margin_bottom = 0
        pf = tf_f.paragraphs[0]
        pf.text = "CarePulse Analytics Platform  •  Hospital Bed Occupancy Forecasting"
        pf.font.size = Pt(9.5)
        pf.font.color.rgb = COLOR_MUTED

        num_box = slide.shapes.add_textbox(Inches(10.5), Inches(7.13), Inches(2.0), Inches(0.3))
        tf_n = num_box.text_frame
        tf_n.margin_left = tf_n.margin_top = tf_n.margin_right = tf_n.margin_bottom = 0
        pn = tf_n.paragraphs[0]
        pn.text = f"Slide {slide_num} of 7"
        pn.font.size = Pt(9.5)
        pn.font.color.rgb = COLOR_MUTED
        pn.alignment = PP_ALIGN.RIGHT


def add_card(slide, left, top, width, height, title, items, border_color=COLOR_CARD_BORDER, bg_color=COLOR_CARD_BG, tag=None, tag_color=COLOR_PURPLE):
    """Adds a modern structured content card."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1.2)

    # Content Box inside the card
    pad_h = Inches(0.25)
    pad_v = Inches(0.22)
    tb = slide.shapes.add_textbox(left + pad_h, top + pad_v, width - (pad_h * 2), height - (pad_v * 2))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    # Optional top tag
    first_para = True
    if tag:
        p_tag = tf.paragraphs[0]
        p_tag.text = tag.upper()
        p_tag.font.size = Pt(9.5)
        p_tag.font.bold = True
        p_tag.font.color.rgb = tag_color
        p_tag.space_after = Pt(4)
        first_para = False

    # Title
    p_title = tf.paragraphs[0] if first_para else tf.add_paragraph()
    p_title.text = title
    p_title.font.size = Pt(15)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_NAVY
    p_title.space_after = Pt(10)

    # Bullet items
    for item in items:
        p = tf.add_paragraph()
        if isinstance(item, tuple):
            lead, body = item
            run_lead = p.add_run()
            run_lead.text = f"•  {lead}: "
            run_lead.font.bold = True
            run_lead.font.size = Pt(11.5)
            run_lead.font.color.rgb = COLOR_NAVY

            run_body = p.add_run()
            run_body.text = body
            run_body.font.size = Pt(11.5)
            run_body.font.color.rgb = COLOR_BODY
        else:
            run = p.add_run()
            run.text = f"•  {item}"
            run.font.size = Pt(11.5)
            run.font.color.rgb = COLOR_BODY
        p.space_after = Pt(6)


# ==============================================================================
# SLIDE 1: TITLE SLIDE
# ==============================================================================
slide1 = prs.slides.add_slide(blank_slide_layout)

# Top decorative accent banner
top_banner = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
top_banner.fill.solid()
top_banner.fill.fore_color.rgb = COLOR_PURPLE
top_banner.line.fill.background()

# Title Platform Badge
sys_badge = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.0), Inches(3.2), Inches(0.4))
sys_badge.fill.solid()
sys_badge.fill.fore_color.rgb = COLOR_PILL_BG
sys_badge.line.color.rgb = COLOR_PURPLE
sys_badge.line.width = Pt(1)
tf_sb = sys_badge.text_frame
p_sb = tf_sb.paragraphs[0]
p_sb.text = "CAREPULSE ANALYTICS PLATFORM"
p_sb.font.size = Pt(10.5)
p_sb.font.bold = True
p_sb.font.color.rgb = COLOR_PILL_TXT
p_sb.alignment = PP_ALIGN.CENTER

# Main Title
title_box = slide1.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(11.3), Inches(1.3))
tf_t = title_box.text_frame
tf_t.word_wrap = True
tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
p1 = tf_t.paragraphs[0]
p1.text = "Hospital Bed Occupancy Forecasting System"
p1.font.size = Pt(32)
p1.font.bold = True
p1.font.color.rgb = COLOR_NAVY
p1.space_after = Pt(8)

p2 = tf_t.add_paragraph()
p2.text = "Predictive Healthcare Resource Planning, Surge Modeling & Conversational Intelligence"
p2.font.size = Pt(16)
p2.font.color.rgb = COLOR_PURPLE

# Separator bar
sep = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(3.1), Inches(11.3), Inches(0.02))
sep.fill.solid()
sep.fill.fore_color.rgb = COLOR_CARD_BORDER
sep.line.fill.background()

# Metadata Details Box (Team & Faculty Placeholders)
left_box = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(3.4), Inches(5.4), Inches(3.4))
left_box.fill.solid()
left_box.fill.fore_color.rgb = COLOR_CARD_BG
left_box.line.color.rgb = COLOR_CARD_BORDER
left_box.line.width = Pt(1)

tf_l = left_box.text_frame
tf_l.word_wrap = True
tf_l.margin_left = tf_l.margin_right = Inches(0.3)
tf_l.margin_top = Inches(0.25)
p_l_tag = tf_l.paragraphs[0]
p_l_tag.text = "PROJECT TEAM DETAILS"
p_l_tag.font.size = Pt(10)
p_l_tag.font.bold = True
p_l_tag.font.color.rgb = COLOR_PURPLE
p_l_tag.alignment = PP_ALIGN.LEFT
p_l_tag.space_after = Pt(8)

p_l_hdr = tf_l.add_paragraph()
p_l_hdr.text = "Presented By:"
p_l_hdr.font.size = Pt(14)
p_l_hdr.font.bold = True
p_l_hdr.font.color.rgb = COLOR_NAVY
p_l_hdr.space_after = Pt(8)

members = [
    "Student / Team Member 1: [Full Name / Roll No / ID]",
    "Student / Team Member 2: [Full Name / Roll No / ID]",
    "Student / Team Member 3: [Full Name / Roll No / ID]",
    "Student / Team Member 4: [Full Name / Roll No / ID]"
]
for m in members:
    p = tf_l.add_paragraph()
    p.text = f"•  {m}"
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_BODY
    p.space_after = Pt(4)

# Right Box: Academic Guidance & Institution
right_box = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(3.4), Inches(5.4), Inches(3.4))
right_box.fill.solid()
right_box.fill.fore_color.rgb = COLOR_CARD_BG
right_box.line.color.rgb = COLOR_CARD_BORDER
right_box.line.width = Pt(1)

tf_r = right_box.text_frame
tf_r.word_wrap = True
tf_r.margin_left = tf_r.margin_right = Inches(0.3)
tf_r.margin_top = Inches(0.25)
p_r_tag = tf_r.paragraphs[0]
p_r_tag.text = "INSTITUTION & MENTORSHIP"
p_r_tag.font.size = Pt(10)
p_r_tag.font.bold = True
p_r_tag.font.color.rgb = COLOR_INDIGO
p_r_tag.alignment = PP_ALIGN.LEFT
p_r_tag.space_after = Pt(8)

p_r_hdr = tf_r.add_paragraph()
p_r_hdr.text = "Academic Affiliation:"
p_r_hdr.font.size = Pt(14)
p_r_hdr.font.bold = True
p_r_hdr.font.color.rgb = COLOR_NAVY
p_r_hdr.space_after = Pt(8)

affils = [
    ("Project Guide", "[Prof. / Dr. Guide Name, Designation]"),
    ("Department", "[Department of Computer Science & Engineering / AIML]"),
    ("Institution", "[University / College / Institute Name]"),
    ("Academic Session", "[Academic Term 2024–2025 / Current Date]")
]
for role, val in affils:
    p = tf_r.add_paragraph()
    r1 = p.add_run()
    r1.text = f"•  {role}: "
    r1.font.bold = True
    r1.font.size = Pt(11)
    r1.font.color.rgb = COLOR_NAVY
    r2 = p.add_run()
    r2.text = val
    r2.font.size = Pt(11)
    r2.font.color.rgb = COLOR_BODY
    p.space_after = Pt(6)


# ==============================================================================
# SLIDE 2: PROBLEM STATEMENT & CLINICAL OBJECTIVES
# ==============================================================================
slide2 = prs.slides.add_slide(blank_slide_layout)
add_header(
    slide2,
    "Clinical Motivation & Objectives",
    "Healthcare Capacity Volatility & Core Objectives",
    "Shifting hospital operations from reactive emergency crisis management to proactive predictive planning",
    slide_num=2
)

# 3 Horizontal Cards
card_w = Inches(3.64)
card_h = Inches(5.1)
top_pos = Inches(1.95)

add_card(
    slide2, Inches(0.8), top_pos, card_w, card_h,
    "The Clinical Dilemma",
    [
        ("Unexpected Bed Surges", "Emergency admissions and seasonal outbreaks trigger unpredictable bed shortages."),
        ("Emergency Boarding", "Admitted patients remain in ER hallways due to inpatient bed unavailability, increasing medical risk."),
        ("Ambulance Diversion", "Critical bottlenecks force emergency diversions to distant healthcare facilities."),
        ("Reactive Resource Allocation", "Traditional administration relies on lagging manual bed counts rather than forward-looking demand signals.")
    ],
    border_color=COLOR_CRIMSON,
    tag="Operational Challenges",
    tag_color=COLOR_CRIMSON
)

add_card(
    slide2, Inches(4.83), top_pos, card_w, card_h,
    "Primary Objectives",
    [
        ("Forecast Bed Occupancy", "Predict continuous daily bed occupancy rates across 8 hospital departments."),
        ("Capacity Planning", "Enable proactive nurse staffing schedules, inventory stocking, and bed preparation."),
        ("Reduce Waiting Times", "Eliminate patient boarding bottlenecks and minimize elective surgery cancellations."),
        ("Identify Peak Demand", "Generate early automated alerts whenever occupancy approaches dangerous capacity thresholds (>85%).")
    ],
    border_color=COLOR_PURPLE,
    tag="Target Goals",
    tag_color=COLOR_PURPLE
)

add_card(
    slide2, Inches(8.86), top_pos, card_w, card_h,
    "Expected Healthcare Impact",
    [
        ("Clinical Efficiency", "Equip medical directors with accurate forecasting windows for synchronized bed turnover."),
        ("Resource Optimization", "Align staffing rosters with anticipated surges, preventing clinical staff burnout."),
        ("Transparent Decision-Making", "Provide conversational query access for non-technical clinicians and managers."),
        ("Evidence-Based Care", "Ensure zero patients are turned away during high-acuity seasonal illness periods.")
    ],
    border_color=COLOR_EMERALD,
    tag="Measurable Outcomes",
    tag_color=COLOR_EMERALD
)


# ==============================================================================
# SLIDE 3: DATASET ARCHITECTURE & CLINICAL FEATURES
# ==============================================================================
slide3 = prs.slides.add_slide(blank_slide_layout)
add_header(
    slide3,
    "Data Architecture",
    "Multi-Department Hospital Time-Series Dataset",
    "Verified clinical records spanning admissions, emergencies, departmental capacities, and seasonal dynamics",
    slide_num=3
)

card_w2 = Inches(5.6)
card_h2 = Inches(2.45)

add_card(
    slide3, Inches(0.8), Inches(1.95), card_w2, card_h2,
    "Dataset Scope & Hygiene",
    [
        ("Data Foundation", "hospital_occupancy.csv featuring 8,768 daily records over a 3-year timeline (2022–2024)."),
        ("8 Clinical Departments", "ICU, Cardiology, Neurology, Pediatrics, Orthopedics, General Medicine, Oncology, Surgery."),
        ("Data Hygiene", "Zero null or missing records; verified temporal continuity and realistic seasonal trends.")
    ],
    tag="Data Profile",
    tag_color=COLOR_INDIGO
)

add_card(
    slide3, Inches(6.8), Inches(1.95), card_w2, card_h2,
    "Patient Flow Features",
    [
        ("Admission Counts", "Daily patient inflow categorized by department and acuity level."),
        ("Emergency Volumes", "Unplanned urgent arrivals driving sudden acute capacity demand spikes."),
        ("Discharge Records", "Planned vs. unplanned patient turnover establishing net bed clearance rates.")
    ],
    tag="Flow Dynamics",
    tag_color=COLOR_EMERALD
)

add_card(
    slide3, Inches(0.8), Inches(4.6), card_w2, card_h2,
    "Clinical & Disease Classification",
    [
        ("Admission Types", "Categorized into Emergency, Urgent, and Elective surgical cohorts."),
        ("Disease Clusters", "Respiratory infections, cardiovascular emergencies, trauma, and chronic disease cohorts."),
        ("Acuity Variations", "Distinguishes high-stay intensive care (ICU) from high-turnover general units.")
    ],
    tag="Clinical Attributes",
    tag_color=COLOR_PURPLE
)

add_card(
    slide3, Inches(6.8), Inches(4.6), card_w2, card_h2,
    "Temporal & Seasonal Indicators",
    [
        ("Seasonal Outbreaks", "Tracks winter viral surges, monsoon illnesses, and summer baseline shifts."),
        ("Weekly Cycles", "Captures weekend admission drops and midweek elective scheduling surges."),
        ("Holiday Surge Factors", "Monitors reduced staffing periods coupled with holiday trauma case spikes.")
    ],
    tag="Temporal Factors",
    tag_color=COLOR_AMBER
)


# ==============================================================================
# SLIDE 4: MACHINE LEARNING PIPELINE & EVALUATION
# ==============================================================================
slide4 = prs.slides.add_slide(blank_slide_layout)
add_header(
    slide4,
    "Machine Learning Pipeline",
    "Model Progression, Feature Engineering & Validation",
    "Systematic evaluation across baseline, intermediate, and advanced predictive architectures",
    slide_num=4
)

add_card(
    slide4, Inches(0.8), top_pos, card_w, card_h,
    "Pipeline & Engineering",
    [
        ("Data Preprocessing", "One-hot encoding of departmental and disease types; robust feature scaling."),
        ("Net Patient Flow", "Computed dynamic feature: Net Flow = Admissions − Discharges."),
        ("Rolling Momentum", "14-day rolling averages and temporal lag indicators to capture admission velocity."),
        ("Cross-Validation", "Chronological temporal train-test split preventing future data leakage.")
    ],
    tag="Feature Engineering",
    tag_color=COLOR_INDIGO
)

add_card(
    slide4, Inches(4.83), top_pos, card_w, card_h,
    "Model Hierarchy",
    [
        ("Baseline: Linear Regression", "Establishes transparent parametric relationships and baseline error benchmarks."),
        ("Intermediate: Random Forest", "Non-linear ensemble capturing complex multi-department and seasonal interactions."),
        ("Advanced: Gradient Boosting", "XGBoost and tree ensembles penalizing outlier variance and sequential trends."),
        ("Model Selection", "Automatically deployed the top-performing validated model pipeline in production.")
    ],
    tag="Modeling Architecture",
    tag_color=COLOR_PURPLE
)

add_card(
    slide4, Inches(8.86), top_pos, card_w, card_h,
    "Evaluation & Outputs",
    [
        ("RMSE & MAE", "Monitored absolute bed occupancy variance ensuring minimal error deviations."),
        ("R² Score", "Evaluated proportion of variance explained across unseen test periods."),
        ("Continuous Occupancy (%)", "Precision forecast of utilization percentage for each department."),
        ("Bed Availability & Alerts", "Direct calculation of available beds and automated peak surge warnings (>85%).")
    ],
    tag="Metrics & Outputs",
    tag_color=COLOR_EMERALD
)


# ==============================================================================
# SLIDE 5: SYSTEM ARCHITECTURE & CAREPULSE ASSISTANT
# ==============================================================================
slide5 = prs.slides.add_slide(blank_slide_layout)
add_header(
    slide5,
    "Full-Stack Implementation",
    "Flask Backend & CarePulse Conversational Assistant",
    "Eliminating dashboard clutter with direct conversational intelligence and verified REST APIs",
    slide_num=5
)

add_card(
    slide5, Inches(0.8), top_pos, card_w, card_h,
    "Flask Backend Engine",
    [
        ("Production Server", "Lightweight Python Flask microservice serving core forecasting and data APIs."),
        ("REST Endpoints", "Structured endpoints for /api/forecast, /api/chat, /api/summary, and /api/chart/<id>."),
        ("Model Inference", "Serialized pipeline (best_model.pkl) executing sub-millisecond predictions."),
        ("Automated Test Suite", "Comprehensive API verification suite: 14 / 14 tests passing (100% success rate).")
    ],
    tag="Backend Architecture",
    tag_color=COLOR_NAVY
)

add_card(
    slide5, Inches(4.83), top_pos, card_w, card_h,
    "Conversational Interface",
    [
        ("Clinician-Centric UX", "Replaces complex, confusing dashboards with a natural language query interface."),
        ("Supported Queries", "Answers questions on overall occupancy, department demand, available beds, and surges."),
        ("Zero Fabrication", "Strictly grounded in real dataset records and verified models—no external paid LLMs."),
        ("Reliability Guarantees", "Transparently informs clinicians when data cannot answer a query reliably.")
    ],
    tag="CarePulse Assistant",
    tag_color=COLOR_PURPLE
)

add_card(
    slide5, Inches(8.86), top_pos, card_w, card_h,
    "Design & Security Standards",
    [
        ("Distraction-Free UI", "Modern professional dark theme with violet accents and crisp readable typography."),
        ("Zero Model Jargon", "Hides technical mathematical equations and model names from hospital staff."),
        ("Strict Data Privacy", "Self-contained local processing with no external API data leaks."),
        ("Instantaneous Navigation", "Seamless page switching between conversational assistant and visual analytics.")
    ],
    tag="UX & Governance",
    tag_color=COLOR_EMERALD
)


# ==============================================================================
# SLIDE 6: GRAPHICAL ANALYSIS & CLINICAL VISUALIZATIONS
# ==============================================================================
slide6 = prs.slides.add_slide(blank_slide_layout)
add_header(
    slide6,
    "Data Intelligence",
    "High-Resolution Visual Analytics Suite",
    "Five purpose-built Matplotlib visualizations with dedicated color palettes and sub-millisecond caching",
    slide_num=6
)

# 2 Columns: Left 3 visual cards, Right 2 visual cards + caching callout
card_w3 = Inches(5.6)
card_h3 = Inches(1.58)

add_card(
    slide6, Inches(0.8), Inches(1.95), card_w3, card_h3,
    "1. Historical Bed Occupancy",
    [
        ("Palette & Features", "Purple & lavender 14-day rolling curve with shaded confidence and 85% peak-occupancy safety alert line.")
    ],
    tag="Macro Trend",
    tag_color=COLOR_PURPLE
)

add_card(
    slide6, Inches(0.8), Inches(3.7), card_w3, card_h3,
    "2. Department-Wise Occupancy",
    [
        ("Palette & Features", "Emerald green, teal, and violet horizontal comparison isolating high-utilization units (ICU, Cardiology).")
    ],
    tag="Department Breakdown",
    tag_color=COLOR_EMERALD
)

add_card(
    slide6, Inches(0.8), Inches(5.45), card_w3, card_h3,
    "3. Inflow vs. Outflow Dynamics",
    [
        ("Palette & Features", "Orange (Admissions) vs. Green (Discharges) dual-curve tracking net hospital accumulation.")
    ],
    tag="Patient Flow",
    tag_color=COLOR_AMBER
)

add_card(
    slide6, Inches(6.8), Inches(1.95), card_w3, card_h3,
    "4. Emergency Surges & Alerts",
    [
        ("Palette & Features", "Crimson daily bars overlaid with a coral 14-day surge curve and a 90th percentile alert threshold.")
    ],
    tag="Acute Surge Detection",
    tag_color=COLOR_CRIMSON
)

add_card(
    slide6, Inches(6.8), Inches(3.7), card_w3, card_h3,
    "5. Model Accuracy Validation",
    [
        ("Palette & Features", "Sunset orange scatter points mapped against an ideal 1:1 purple diagonal line with a ±5% tolerance zone.")
    ],
    tag="Predictive Validation",
    tag_color=COLOR_INDIGO
)

add_card(
    slide6, Inches(6.8), Inches(5.45), card_w3, card_h3,
    "Performance & Rendering Reliability",
    [
        ("In-Memory Caching", "Thread-safe PNG cache delivering sub-millisecond chart loading with zero freeze states.")
    ],
    tag="Engineering Optimization",
    tag_color=COLOR_NAVY
)


# ==============================================================================
# SLIDE 7: OPERATIONAL IMPACT & FUTURE ROADMAP
# ==============================================================================
slide7 = prs.slides.add_slide(blank_slide_layout)
add_header(
    slide7,
    "Impact & Future Scope",
    "Healthcare Transformation & Scalability Roadmap",
    "Delivering measurable clinical efficiencies and extending predictive intelligence across regional networks",
    slide_num=7
)

add_card(
    slide7, Inches(0.8), top_pos, card_w, card_h,
    "Operational Hospital Value",
    [
        ("Surge Preparedness", "Shifts management from reactive crisis response to proactive early-warning surge mitigation."),
        ("Optimized Staffing", "Harmonizes nursing shift rosters with predicted patient loads, reducing staff burnout."),
        ("Surgical Flow Protection", "Minimizes last-minute elective surgery cancellations through anticipated bed availability."),
        ("Reduced Boarding", "Decreases emergency room boarding times, accelerating clinical triage and care delivery.")
    ],
    tag="Clinical Benefits",
    tag_color=COLOR_EMERALD
)

add_card(
    slide7, Inches(4.83), top_pos, card_w, card_h,
    "Technical Accomplishments",
    [
        ("Full-Stack Integration", "Flawless coupling of verified ML inference, Flask microservice, and dynamic frontend."),
        ("Conversational Intelligence", "Democratizes complex predictive data into natural language answers for hospital staff."),
        ("100% Test Coverage", "Automated test suite (14/14 passed) validating endpoints, forecasting logic, and chart rendering."),
        ("Robust Data Integrity", "Zero data fabrication; models trained and validated strictly on authentic clinical records.")
    ],
    tag="Core Milestones",
    tag_color=COLOR_PURPLE
)

add_card(
    slide7, Inches(8.86), top_pos, card_w, card_h,
    "Future Expansion Roadmap",
    [
        ("Real-Time EHR Integration", "Direct FHIR / HL7 standard data ingestion from electronic health record systems."),
        ("Regional Bed Federation", "Multi-hospital network coordination enabling regional load-balancing and smart ambulance routing."),
        ("Sub-Specialty Triage", "Granular ward-level predictions (e.g., Pediatric ICU, Cardiac Care Units)."),
        ("Automated Staffing Rosters", "Algorithmic nurse-to-patient allocation directly linked to forecasted bed demand.")
    ],
    tag="Future Horizons",
    tag_color=COLOR_INDIGO
)

# Output Presentation Path
output_pptx = r"C:\Users\LENOVO\OneDrive\Desktop\AIML Project\Hospital_Bed_Occupancy_Forecasting.pptx"
prs.save(output_pptx)
print(f"Presentation successfully created and saved to: {output_pptx}")
