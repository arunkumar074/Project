import os
import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
from gtts import gTTS
import base64
import json
import requests
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found. Please set it in Streamlit Secrets.")
    st.stop()

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}")
    st.stop()

LANG_OPTIONS = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Hindi (हिंदी)": "hi",
    "Telugu (తెలుగు)": "te",
    "Tamil (தமிழ்)": "ta",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Marathi (मराठी)": "mr",
    "Bengali (বাংলা)": "bn"
}

LANG_PLAIN_NAME = {
    "English": "English",
    "Spanish": "Spanish",
    "French": "French",
    "Hindi (हिंदी)": "Hindi",
    "Telugu (తెలుగు)": "Telugu",
    "Tamil (தமிழ்)": "Tamil",
    "Kannada (ಕನ್ನಡ)": "Kannada",
    "Marathi (मराठी)": "Marathi",
    "Bengali (বাংলা)": "Bengali"
}

# -----------------------------------------------------------------------
# PHARMACEUTICAL DATABASE INTEGRATION (NEW)
# Uses: RxNorm (NLM) + OpenFDA — both FREE, no API key needed
# -----------------------------------------------------------------------

def get_rxcui(medicine_name):
    """
    Resolve a medicine name to an RxNorm concept identifier (RxCUI).
    RxCUI is the standard ID used to look up drug interactions.
    Returns RxCUI string or None if not found.
    """
    try:
        # Try approximate match first for brand names
        url = f"https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
        params = {"term": medicine_name, "maxEntries": 1, "option": 1}
        resp = requests.get(url, params=params, timeout=8)
        data = resp.json()
        candidates = data.get("approximateGroup", {}).get("candidate", [])
        if candidates:
            return candidates[0].get("rxcui")

        # Fallback: exact spelling search
        url2 = f"https://rxnav.nlm.nih.gov/REST/rxcui.json"
        params2 = {"name": medicine_name, "search": 2}
        resp2 = requests.get(url2, params=params2, timeout=8)
        data2 = resp2.json()
        id_group = data2.get("idGroup", {})
        rxcuis = id_group.get("rxnormId", [])
        return rxcuis[0] if rxcuis else None
    except Exception:
        return None


def get_drug_interactions_rxnorm(rxcui_list):
    """
    Query NLM's RxNorm DDI (Drug-Drug Interaction) API with a list of RxCUIs.
    Returns a list of interaction dicts: {medicine_pair, severity, description}.
    This database is sourced from DrugBank and the ONC High-Priority DDI list.
    """
    interactions = []
    if len(rxcui_list) < 2:
        return interactions
    try:
        rxcuis_str = "+".join([r for r in rxcui_list if r])
        url = f"https://rxnav.nlm.nih.gov/REST/interaction/list.json"
        params = {"rxcuis": rxcuis_str}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        full_interaction_type_group = data.get("fullInteractionTypeGroup", [])
        for group in full_interaction_type_group:
            source = group.get("sourceName", "")
            for interaction_type in group.get("fullInteractionType", []):
                for pair in interaction_type.get("interactionPair", []):
                    concepts = pair.get("interactionConcept", [])
                    if len(concepts) >= 2:
                        drug1 = concepts[0].get("minConceptItem", {}).get("name", "")
                        drug2 = concepts[1].get("minConceptItem", {}).get("name", "")
                        severity = pair.get("severity", "unknown")
                        description = pair.get("description", "")
                        interactions.append({
                            "drug1": drug1,
                            "drug2": drug2,
                            "severity": severity,
                            "description": description,
                            "source": source
                        })
    except Exception:
        pass
    return interactions


def get_openfda_drug_info(medicine_name):
    """
    Query OpenFDA for official FDA-approved labeling data:
    warnings, boxed warnings, and adverse reactions.
    This is real pharmaceutical label data, not AI-generated.
    """
    result = {
        "boxed_warning": None,
        "warnings": None,
        "adverse_reactions": None,
        "found": False
    }
    try:
        url = "https://api.fda.gov/drug/label.json"
        params = {
            "search": f'openfda.brand_name:"{medicine_name}" OR openfda.generic_name:"{medicine_name}"',
            "limit": 1
        }
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                label = results[0]
                result["found"] = True
                # Boxed warning (most serious FDA warning)
                bw = label.get("boxed_warning", [])
                result["boxed_warning"] = bw[0][:500] if bw else None
                # General warnings
                w = label.get("warnings", [])
                result["warnings"] = w[0][:400] if w else None
                # Adverse reactions
                ar = label.get("adverse_reactions", [])
                result["adverse_reactions"] = ar[0][:400] if ar else None
    except Exception:
        pass
    return result


def verify_medicines_with_pharma_db(medicine_names):
    """
    Master function: for a list of medicine names,
    1. Resolve each to an RxCUI via RxNorm
    2. Check all drug-drug interactions via NLM DDI API (DrugBank-backed)
    3. Fetch FDA label warnings via OpenFDA
    Returns a structured dict with all findings.
    """
    verification = {
        "rxcui_map": {},          # medicine_name -> rxcui
        "interactions": [],        # list of DDI findings from real DB
        "fda_warnings": {},        # medicine_name -> FDA label info
        "unresolved": []           # medicines not found in RxNorm
    }

    # Step 1: Resolve RxCUIs
    rxcui_list = []
    for name in medicine_names:
        rxcui = get_rxcui(name)
        if rxcui:
            verification["rxcui_map"][name] = rxcui
            rxcui_list.append(rxcui)
        else:
            verification["unresolved"].append(name)
        time.sleep(0.2)  # Respect NLM rate limits

    # Step 2: Drug-drug interactions from real database
    if len(rxcui_list) >= 2:
        interactions = get_drug_interactions_rxnorm(rxcui_list)
        verification["interactions"] = interactions

    # Step 3: FDA label warnings per medicine
    for name in medicine_names:
        fda_info = get_openfda_drug_info(name)
        if fda_info["found"]:
            verification["fda_warnings"][name] = fda_info
        time.sleep(0.2)  # Respect FDA rate limits

    return verification


def format_pharma_verification_block(verification, target_lang):
    """
    Format the pharmaceutical database verification results
    into a clean markdown block for display.
    """
    md = "---\n"
    md += "## 🏥 Pharmaceutical Database Verification\n"
    md += "*Source: NLM RxNorm + DrugBank DDI Database + OpenFDA Official Labels*\n\n"

    # RxNorm resolution status
    md += "### 🔍 Medicine Database Lookup (RxNorm)\n"
    if verification["rxcui_map"]:
        for name, rxcui in verification["rxcui_map"].items():
            md += f"✅ **{name}** — Found in RxNorm (ID: `{rxcui}`)\n"
    if verification["unresolved"]:
        for name in verification["unresolved"]:
            md += f"⚠️ **{name}** — Not found in RxNorm (may be Indian brand name; AI analysis used instead)\n"
    md += "\n"

    # Real DDI findings
    md += "### ⚡ Drug-Drug Interactions (DrugBank via NLM)\n"
    if verification["interactions"]:
        for itn in verification["interactions"]:
            severity = itn.get("severity", "unknown").upper()
            drug1 = itn.get("drug1", "")
            drug2 = itn.get("drug2", "")
            desc = itn.get("description", "")
            source = itn.get("source", "")
            if severity in ["HIGH", "N/A"] or "contraindicated" in desc.lower():
                icon = "🚨"
            elif severity == "MODERATE":
                icon = "⚠️"
            else:
                icon = "ℹ️"
            md += f"{icon} **{severity}** — {drug1} ↔ {drug2}\n"
            if desc:
                md += f"   > {desc[:300]}\n"
            md += "\n"
    else:
        if len(verification["rxcui_map"]) >= 2:
            md += "✅ No known drug-drug interactions found in DrugBank/NLM database for resolved medicines.\n\n"
        else:
            md += "ℹ️ Interaction check requires at least 2 medicines to be resolved in RxNorm.\n\n"

    # FDA label warnings
    md += "### 📋 FDA Official Label Warnings (OpenFDA)\n"
    if verification["fda_warnings"]:
        for name, fda_info in verification["fda_warnings"].items():
            md += f"**{name}:**\n"
            if fda_info.get("boxed_warning"):
                md += f"🚨 **FDA Boxed Warning:** {fda_info['boxed_warning'][:300]}...\n\n"
            if fda_info.get("warnings"):
                md += f"⚠️ **Warnings:** {fda_info['warnings'][:250]}...\n\n"
            if not fda_info.get("boxed_warning") and not fda_info.get("warnings"):
                md += "✅ No boxed warnings found in FDA label database.\n\n"
    else:
        md += "ℹ️ FDA label data not found for these medicines (common for Indian brand names not registered with FDA).\n\n"

    md += "> 🔬 *This verification uses live pharmaceutical databases. Always consult your doctor or pharmacist.*\n"
    md += "---\n\n"
    return md


# -----------------------------------------------------------------------
# CORE: Get medicine info AND translate in ONE single Gemini call
# -----------------------------------------------------------------------

def get_medicine_info_from_gemini(medicine_name, all_medicines, patient_allergies, target_lang, pharma_verification=None):
    """
    Single Gemini API call that:
    1. Analyzes the medicine clinically
    2. Returns ALL text already in the target language
    3. Now also cross-references real pharmaceutical DB findings (NEW)
    """
    other_medicines = [m for m in all_medicines if m.lower() != medicine_name.lower()]
    other_meds_str = ", ".join(other_medicines) if other_medicines else "None"

    # NEW: Build pharma DB context to inject into Gemini prompt
    pharma_context = ""
    if pharma_verification:
        # Inject real DDI findings
        relevant_interactions = [
            i for i in pharma_verification.get("interactions", [])
            if medicine_name.lower() in i.get("drug1", "").lower()
            or medicine_name.lower() in i.get("drug2", "").lower()
        ]
        if relevant_interactions:
            pharma_context += "\n\nVERIFIED DRUG INTERACTIONS FROM DRUGBANK/NLM DATABASE (use these as ground truth):\n"
            for itn in relevant_interactions:
                pharma_context += (
                    f"- {itn['drug1']} ↔ {itn['drug2']}: "
                    f"Severity={itn['severity']}, {itn['description'][:200]}\n"
                )

        # Inject FDA boxed warnings
        fda_info = pharma_verification.get("fda_warnings", {}).get(medicine_name, {})
        if fda_info.get("boxed_warning"):
            pharma_context += f"\nFDA BOXED WARNING (official): {fda_info['boxed_warning'][:300]}\n"
        if fda_info.get("warnings"):
            pharma_context += f"\nFDA WARNINGS (official): {fda_info['warnings'][:250]}\n"

    if target_lang == "English":
        lang_instruction = "Write ALL text values in English."
    else:
        lang_instruction = (
            f"IMPORTANT: Write ALL text values (usage, side_effects, general_warnings, "
            f"interaction notes, allergy message, overall_interaction_summary) in {target_lang} language. "
            f"You MUST use {target_lang} script. Do NOT write in English for these fields. "
            f"Only keep medicine brand/generic names and drug_class in English."
        )

    prompt = f"""You are an expert clinical pharmacist.

Medicine: {medicine_name}
Other medicines in same prescription: {other_meds_str}
Patient allergies: {patient_allergies or "None"}
{pharma_context}

{lang_instruction}

IMPORTANT: If verified drug interactions are provided above from the pharmaceutical database,
prioritize those over your own knowledge. Mark those interactions as database-verified.

Return ONLY this valid JSON, no markdown, no extra text:

{{
  "usage": [
    "first use or condition treated --- in {target_lang}",
    "mechanism of action --- in {target_lang}",
    "another key benefit --- in {target_lang}"
  ],
  "side_effects": [
    "side effect 1 in {target_lang}",
    "side effect 2 in {target_lang}",
    "side effect 3 in {target_lang}",
    "side effect 4 in {target_lang}"
  ],
  "drug_class": "pharmacological class in English only",
  "interaction_with_prescribed": [
    {{
      "medicine": "other medicine name in English",
      "safe": true,
      "note": "interaction note in {target_lang}",
      "db_verified": false
    }}
  ],
  "general_warnings": [
    "warning 1 in {target_lang}",
    "warning 2 in {target_lang}"
  ],
  "allergy_alert": {{
    "triggered": false,
    "message": "allergy message in {target_lang} if triggered, else empty string"
  }},
  "overall_interaction_summary": "one line summary in {target_lang}"
}}

Rules:
- Set allergy_alert.triggered = true ONLY if patient allergy matches drug class of {medicine_name}
- Set safe = false if known moderate or major interaction exists
- Set db_verified = true for interactions confirmed by the pharmaceutical database above
- Return ONLY valid JSON"""

    fallback = {
        "usage": ["Information not available"],
        "side_effects": ["Information not available"],
        "drug_class": "Unknown",
        "interaction_with_prescribed": [],
        "general_warnings": [],
        "allergy_alert": {"triggered": False, "message": ""},
        "overall_interaction_summary": "Could not retrieve data."
    }

    try:
        response = model.generate_content(prompt)
        json_str = response.text.strip()
        json_str = json_str.replace("```json", "").replace("```", "").strip()
        start = json_str.find("{")
        end = json_str.rfind("}") + 1
        if start != -1 and end > start:
            json_str = json_str[start:end]
        return json.loads(json_str)
    except Exception:
        return fallback


# -----------------------------------------------------------------------
# Translate dosage fields — single call for all fields at once
# -----------------------------------------------------------------------

def translate_dosage_fields(prescription_list, target_lang):
    if target_lang == "English":
        return prescription_list

    items_to_translate = []
    for i, item in enumerate(prescription_list):
        freq = item.get("Frequency/Instructions", "")
        dosage = item.get("Dosage Details", "")
        if freq:
            items_to_translate.append({"idx": i, "field": "Frequency/Instructions", "text": freq})
        if dosage:
            items_to_translate.append({"idx": i, "field": "Dosage Details", "text": dosage})

    if not items_to_translate:
        return prescription_list

    numbered = "\n".join([f"{j+1}. {t['text']}" for j, t in enumerate(items_to_translate)])
    prompt = f"""Translate each numbered text below into {target_lang} language.

RULES:
- You MUST translate into {target_lang}. Do NOT return English text.
- Keep medicine names, numbers, mg values, and patterns like 1-0-1 in English.
- Return ONLY a JSON array with the translated texts in the same order.
- Format: ["translation 1", "translation 2", ...]
- No explanation, no markdown, just the JSON array.

Texts to translate:
{numbered}

JSON array of {target_lang} translations:"""

    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        start = result.find("[")
        end = result.rfind("]") + 1
        if start != -1 and end > start:
            translations = json.loads(result[start:end])
            for j, t in enumerate(items_to_translate):
                if j < len(translations) and translations[j]:
                    prescription_list[t["idx"]][t["field"]] = translations[j]
    except Exception:
        pass
    return prescription_list


# -----------------------------------------------------------------------
# BUILD MEDICINE DISPLAY CARD
# -----------------------------------------------------------------------

def build_medicine_card(idx, med_name, dosage, instructions, pattern, duration, med_info):
    md = f"---\n### 💊 {idx}. {med_name}\n\n"

    # Usage
    md += "✅ **Usage:**\n"
    for point in med_info.get("usage", []):
        point = point.strip().lstrip("-").lstrip("*").strip()
        if point:
            md += f"- {point}\n"
    md += "\n"

    # Dosage
    md += f"💊 **Dosage:** {dosage}\n"
    md += f"- {instructions}\n"
    if pattern:
        md += f"- **({pattern} pattern)**\n"
    if duration:
        md += f"- Duration: **{duration}**\n"
    md += "\n"

    # Side Effects
    md += "⚠️ **Common Side Effects:**\n"
    for effect in med_info.get("side_effects", []):
        effect = effect.strip().lstrip("-").lstrip("*").strip()
        if effect:
            md += f"- {effect}\n"
    md += "\n"

    # Allergy Alert or Drug Interaction
    allergy_info = med_info.get("allergy_alert", {})
    if allergy_info.get("triggered", False):
        drug_class = med_info.get("drug_class", "this drug class")
        md += "🚨 **Drug Interaction & Allergy Check:**\n"
        md += f"❌ **ALERT:** {allergy_info.get('message', '')}\n"
        md += f"⚠ **{med_name}** belongs to **{drug_class}**\n"
        md += "👉 **Recommendation: Consult doctor immediately before taking this medicine.**\n\n"
    else:
        md += "🔎 **Drug Interaction Check:**\n"
        interactions = med_info.get("interaction_with_prescribed", [])
        if interactions:
            for interaction in interactions:
                other_med = interaction.get("medicine", "")
                is_safe = interaction.get("safe", True)
                note = interaction.get("note", "")
                db_verified = interaction.get("db_verified", False)
                verified_badge = " 🏥 *[DB Verified]*" if db_verified else ""
                if is_safe:
                    md += f"✔ Safe with **{other_med}**{verified_badge}"
                    if note:
                        md += f" — {note}"
                    md += "\n"
                else:
                    md += f"⚠ Use cautiously with **{other_med}**{verified_badge} — {note}\n"

        for warning in med_info.get("general_warnings", []):
            warning = warning.strip()
            if warning:
                md += f"⚠ {warning}\n"

        summary = med_info.get("overall_interaction_summary", "")
        if summary:
            md += f"\n📋 *{summary}*\n"

    md += "\n"
    return md


# -----------------------------------------------------------------------
# AUDIO HELPERS
# -----------------------------------------------------------------------

def clean_text_for_speech(text):
    symbols = [
        "*", "#", "|", "`", "_", "---", "**",
        "✅", "⚠️", "⚠", "🔎", "💊", "🚨", "❌",
        "✔", "📋", "👉", "🧾", "👤", "👨", "🔊", "🏥", "🔬"
    ]
    for s in symbols:
        text = text.replace(s, "")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line and line.strip("-").strip()]
    return "\n".join(lines)


def text_to_audio(text, lang="en"):
    try:
        clean_text = clean_text_for_speech(text)
        if not clean_text.strip():
            return "<p>No text to speak.</p>"
        gtts_lang = lang if lang in ["en", "es", "fr", "hi", "ta", "bn", "kn", "mr", "te"] else "en"
        chunk_size = 500
        chunks = []
        while len(clean_text) > chunk_size:
            split_at = clean_text.rfind(".", 0, chunk_size)
            if split_at == -1:
                split_at = clean_text.rfind("\n", 0, chunk_size)
            if split_at == -1:
                split_at = chunk_size
            chunks.append(clean_text[: split_at + 1].strip())
            clean_text = clean_text[split_at + 1:].strip()
        if clean_text:
            chunks.append(clean_text)

        combined = io.BytesIO()
        for chunk in chunks:
            if chunk.strip():
                try:
                    tts = gTTS(text=chunk, lang=gtts_lang)
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    combined.write(fp.read())
                except Exception:
                    continue
        combined.seek(0)
        audio_b64 = base64.b64encode(combined.read()).decode()
        return f'<audio controls src="data:audio/mp3;base64,{audio_b64}">Your browser does not support audio.</audio>'
    except Exception as e:
        return f"<p>Error in voice generation: {e}</p>"


def parse_frequency_to_times(instructions):
    instructions_lower = instructions.lower()
    if "morning" in instructions_lower and "evening" in instructions_lower:
        return ["08:00", "20:00"]
    elif "three times" in instructions_lower or "3 times" in instructions_lower:
        return ["08:00", "14:00", "20:00"]
    elif "four times" in instructions_lower or "4 times" in instructions_lower:
        return ["08:00", "12:00", "16:00", "20:00"]
    elif "twice" in instructions_lower or "two times" in instructions_lower:
        return ["08:00", "20:00"]
    elif "night" in instructions_lower or "bedtime" in instructions_lower:
        return ["21:00"]
    elif "morning" in instructions_lower:
        return ["08:00"]
    else:
        return ["08:00"]


# --- Session State ---
if "reminders" not in st.session_state:
    st.session_state.reminders = []
if "extracted_medicines" not in st.session_state:
    st.session_state.extracted_medicines = {}
if "patient_info_display" not in st.session_state:
    st.session_state.patient_info_display = None
if "medicine_display_text" not in st.session_state:
    st.session_state.medicine_display_text = None
if "full_prescription" not in st.session_state:
    st.session_state.full_prescription = None
if "medicine_cards_list" not in st.session_state:
    st.session_state.medicine_cards_list = []
if "pharma_verification" not in st.session_state:   # NEW
    st.session_state.pharma_verification = None


# -----------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="AI Smart Prescription Guide",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("💊 AI Smart Prescription Guide for Safer Healthcare")
    st.caption("Powered by Gemini AI + RxNorm + DrugBank + OpenFDA Pharmaceutical Databases")

    # --- Sidebar ---
    st.sidebar.header("⚙️ Settings & Features")

    selected_lang_name = st.sidebar.selectbox(
        "🌐 Select Language for Output:", list(LANG_OPTIONS.keys())
    )
    target_lang_code = LANG_OPTIONS[selected_lang_name]
    clean_lang = LANG_PLAIN_NAME[selected_lang_name]

    voice_guidance_enabled = st.sidebar.checkbox("🔊 Enable Voice Guidance", value=True)

    # NEW: Pharma DB toggle
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏥 Pharmaceutical Database")
    pharma_db_enabled = st.sidebar.checkbox(
        "Enable Real Database Verification",
        value=True,
        help="Cross-checks medicines against NLM RxNorm, DrugBank DDI database, and OpenFDA. Adds ~15-30 seconds."
    )
    if pharma_db_enabled:
        st.sidebar.info("✅ RxNorm + DrugBank + OpenFDA active")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⏰ Set Medication Reminders")
    medicine_names = list(st.session_state.extracted_medicines.keys())
    if medicine_names:
        st.sidebar.markdown("**Step 1: Select Medicine**")
        selected_medicine = st.sidebar.selectbox(
            "Select medicine for reminder:",
            options=["-- Select Medicine --"] + medicine_names,
            key="reminder_med_select",
        )
        if selected_medicine and selected_medicine != "-- Select Medicine --":
            med_data = st.session_state.extracted_medicines[selected_medicine]
            st.sidebar.markdown("**Step 2: Set Alarm Times**")
            instructions = med_data.get("Frequency/Instructions", "")
            suggested_times = parse_frequency_to_times(instructions)
            num_alarms = st.sidebar.number_input(
                "Number of alarms per day:",
                min_value=1, max_value=4,
                value=len(suggested_times),
                key="num_alarms_input",
            )
            alarm_times = []
            for i in range(num_alarms):
                default_time_str = suggested_times[i] if i < len(suggested_times) else "08:00"
                hour, minute = map(int, default_time_str.split(":"))
                c1, c2 = st.sidebar.columns(2)
                with c1:
                    alarm_hour = st.number_input(
                        f"Alarm {i+1} Hour:", min_value=0, max_value=23,
                        value=hour, key=f"hour_{selected_medicine}_{i}",
                    )
                with c2:
                    alarm_minute = st.number_input(
                        f"Minute:", min_value=0, max_value=59,
                        value=minute, key=f"minute_{selected_medicine}_{i}",
                    )
                alarm_times.append(f"{alarm_hour:02d}:{alarm_minute:02d}")

            st.sidebar.markdown("**Step 3: Notes**")
            additional_notes = st.sidebar.text_input(
                "Notes (optional):", value=instructions[:50], key="notes_input"
            )
            final_reminder = f"{selected_medicine} | {', '.join(alarm_times)} | {additional_notes}"
            if st.sidebar.button("🔔 Add Reminder with Alarm"):
                if final_reminder not in st.session_state.reminders:
                    st.session_state.reminders.append(final_reminder)
                    st.sidebar.success(f"✅ Reminder added with {len(alarm_times)} alarm(s)!")
                else:
                    st.sidebar.warning("⚠️ This reminder already exists.")
    else:
        st.sidebar.info("📋 Upload and analyze a prescription first to enable reminders.")

    st.sidebar.markdown("---")
    if st.session_state.reminders:
        st.sidebar.subheader("📱 Active Reminders")
        for idx, reminder in enumerate(st.session_state.reminders):
            c1, c2 = st.sidebar.columns([5, 1])
            with c1:
                st.sidebar.text(f"{idx+1}. {reminder}")
            with c2:
                if st.sidebar.button("❌", key=f"del_{idx}"):
                    st.session_state.reminders.pop(idx)
                    st.rerun()

    st.markdown("---")

    # --- Main Content ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload Prescription Image")
        uploaded_file = st.file_uploader(
            "Choose a prescription image", type=["png", "jpg", "jpeg"]
        )
        image = None
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Prescription", use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")

    with col2:
        st.subheader("2. Analyze Prescription")
        if image is not None:
            extraction_prompt = """
You are a medical prescription analyzer.
Extract ALL information from this prescription image.
Keep ALL medicine names and dosage instructions in English.
Return ONLY valid JSON, no other text:

{
    "patient_info": {
        "name": "patient name",
        "age": "age",
        "sex": "sex",
        "date": "date",
        "known_allergies": "allergies mentioned or None"
    },
    "doctor_info": {
        "name": "doctor name",
        "license": "license number or N/A"
    },
    "prescription": [
        {
            "Medicine Name": "medicine brand name with strength in English",
            "Dosage Details": "strength and form e.g. 625mg tablet",
            "Frequency/Instructions": "full dosage instructions in English",
            "Dosage Pattern": "pattern like 1-0-1",
            "Duration": "e.g. 5 days"
        }
    ]
}
"""
            if st.button("🔍 Extract and Analyze"):
                for key in list(st.session_state.keys()):
                    if key.startswith("hour_") or key.startswith("minute_"):
                        del st.session_state[key]

                # STEP 1: Extract prescription from image
                with st.spinner("📖 Reading prescription image..."):
                    try:
                        try:
                            response = model.generate_content(
                                [extraction_prompt, image],
                                generation_config=genai.GenerationConfig(
                                    response_mime_type="application/json"
                                ),
                            )
                        except Exception:
                            response = model.generate_content([extraction_prompt, image])

                        json_str = (
                            response.text.strip()
                            .replace("```json", "")
                            .replace("```", "")
                            .strip()
                        )
                        data = json.loads(json_str)
                        patient = data.get("patient_info", {})
                        doctor = data.get("doctor_info", {})
                        prescription_list = data.get("prescription", [])
                        patient_allergies = patient.get("known_allergies", "None")

                        patient_text = "## 🧾 AI Smart Prescription Guide\n\n"
                        patient_text += "### 👤 Patient Information\n"
                        patient_text += f"- **Name:** {patient.get('name', 'N/A')}\n"
                        patient_text += f"- **Age:** {patient.get('age', 'N/A')}\n"
                        patient_text += f"- **Sex:** {patient.get('sex', 'N/A')}\n"
                        patient_text += f"- **Date:** {patient.get('date', 'N/A')}\n"
                        patient_text += f"- **Known Allergies:** {patient_allergies}\n\n"
                        patient_text += "### 👨‍⚕️ Doctor Information\n"
                        patient_text += f"- **Name:** {doctor.get('name', 'N/A')}\n"
                        patient_text += f"- **License:** {doctor.get('license', 'N/A')}\n\n"
                        st.session_state.patient_info_display = patient_text

                        if clean_lang != "English":
                            with st.spinner(f"🌐 Translating dosage instructions to {clean_lang}..."):
                                prescription_list = translate_dosage_fields(prescription_list, clean_lang)

                        all_medicine_names = []
                        extracted_med_map = {}
                        for item in prescription_list:
                            med_name = item.get("Medicine Name", "N/A")
                            all_medicine_names.append(med_name)
                            extracted_med_map[med_name] = item
                        st.session_state.extracted_medicines = extracted_med_map

                    except json.JSONDecodeError:
                        st.error("Could not parse prescription. Try a clearer image.")
                        st.stop()
                    except Exception as e:
                        st.error(f"Extraction error: {e}")
                        st.stop()

                st.success(f"✅ Found {len(prescription_list)} medicine(s). Fetching AI analysis in {clean_lang}...")

                # STEP 2 (NEW): Pharmaceutical database verification
                pharma_verification = None
                if pharma_db_enabled:
                    with st.spinner("🏥 Cross-verifying with pharmaceutical databases (RxNorm + DrugBank + OpenFDA)..."):
                        pharma_verification = verify_medicines_with_pharma_db(all_medicine_names)
                        st.session_state.pharma_verification = pharma_verification

                    # Show summary of DB verification
                    resolved_count = len(pharma_verification["rxcui_map"])
                    unresolved_count = len(pharma_verification["unresolved"])
                    interaction_count = len(pharma_verification["interactions"])
                    fda_count = len(pharma_verification["fda_warnings"])
                    st.info(
                        f"🏥 Database check complete: "
                        f"{resolved_count} medicine(s) found in RxNorm, "
                        f"{interaction_count} real DDI interaction(s) found, "
                        f"{fda_count} FDA label(s) retrieved. "
                        f"{unresolved_count} medicine(s) not in RxNorm (Indian brands — AI used instead)."
                    )

                # STEP 3: Per-medicine — Gemini analysis (now with pharma DB context)
                medicine_text = ""
                medicine_cards_list = []
                progress_bar = st.progress(0, text="Starting analysis...")
                total = len(prescription_list)

                for idx, item in enumerate(prescription_list, 1):
                    med_name = item.get("Medicine Name", "N/A")
                    dosage = item.get("Dosage Details", "N/A")
                    instructions = item.get("Frequency/Instructions", "N/A")
                    pattern = item.get("Dosage Pattern", "")
                    duration = item.get("Duration", "")

                    progress_bar.progress(
                        int((idx / total) * 100),
                        text=f"🔍 Analyzing {med_name} in {clean_lang} ({idx}/{total})...",
                    )

                    # Pass pharma DB verification into Gemini call
                    med_info = get_medicine_info_from_gemini(
                        medicine_name=med_name,
                        all_medicines=all_medicine_names,
                        patient_allergies=patient_allergies,
                        target_lang=clean_lang,
                        pharma_verification=pharma_verification,   # NEW
                    )

                    card = build_medicine_card(
                        idx, med_name, dosage, instructions, pattern, duration, med_info
                    )
                    medicine_text += card
                    medicine_cards_list.append({"name": med_name, "text": card})

                progress_bar.empty()
                st.session_state.medicine_display_text = medicine_text
                st.session_state.medicine_cards_list = medicine_cards_list
                st.session_state.full_prescription = patient_text + medicine_text
                st.success(f"✅ Full Analysis Complete in {clean_lang}!")

    st.markdown("---")

    # --- Results Display ---
    if st.session_state.full_prescription:
        if st.session_state.patient_info_display:
            st.markdown(st.session_state.patient_info_display)

        # NEW: Show pharmaceutical database verification block
        if st.session_state.pharma_verification:
            pharma_block = format_pharma_verification_block(
                st.session_state.pharma_verification, clean_lang
            )
            st.markdown(pharma_block)

        st.markdown("---")
        st.subheader(f"💊 Prescribed Medicines — Detailed Analysis ({selected_lang_name})")
        if st.session_state.medicine_display_text:
            st.markdown(st.session_state.medicine_display_text)

        if voice_guidance_enabled:
            st.markdown("---")
            st.markdown("### 🔊 Voice Guidance")
            st.markdown("**👤 Patient & Doctor Info:**")
            audio_html = text_to_audio(
                st.session_state.patient_info_display or "", lang=target_lang_code
            )
            st.markdown(audio_html, unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("**💊 Medicine-wise Audio:**")
            med_cards = st.session_state.get("medicine_cards_list", [])
            if med_cards:
                for card_info in med_cards:
                    st.markdown(f"🔊 *{card_info['name']}*")
                    audio_html = text_to_audio(card_info["text"], lang=target_lang_code)
                    st.markdown(audio_html, unsafe_allow_html=True)
            else:
                audio_html = text_to_audio(
                    st.session_state.medicine_display_text or "", lang=target_lang_code
                )
                st.markdown(audio_html, unsafe_allow_html=True)

        st.markdown("---")
        st.warning(
            "⚠️ **Important Disclaimer:** This AI analysis is for informational purposes only. "
            "Always consult your doctor or pharmacist for comprehensive medical advice. "
            "Do not modify your prescription based solely on this report. "
            "Pharmaceutical database data is sourced from NLM/NIH and FDA public databases."
        )


if __name__ == "__main__":
    main()
