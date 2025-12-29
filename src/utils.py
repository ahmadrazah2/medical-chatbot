import re

def clean_msg(text: str) -> str:
    text = text.strip()
    return re.sub(r"\s*\d{1,2}:\d{1,2}\s*$", "", text)


def is_korean(text: str) -> bool:
    return re.search(r"[\uac00-\ud7a3]", text) is not None


def ko_to_en(llm, ko_text: str) -> str:
    prompt = (
        "Translate the following Korean medical question to English.\n"
        "Output ONLY the English translation.\n\n"
        f"Korean: {ko_text}\nEnglish:"
    )
    out = llm.invoke(prompt)
    return out.strip() if isinstance(out, str) else out.content.strip()


def en_to_ko(llm, en_text: str) -> str:
    prompt = (
        "Translate the following medical answer to Korean.\n"
        "Output ONLY Korean.\n\n"
        f"English: {en_text}\nKorean:"
    )
    out = llm.invoke(prompt)
    return out.strip() if isinstance(out, str) else out.content.strip()
