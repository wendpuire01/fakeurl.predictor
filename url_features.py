"""
url_features.py
================
Extraction des 33 features lexicales d'une URL, d'apres l'approche
"Feature Engineering" de Nana et al. (2024), "Characterization of Malicious
URLs Using Machine Learning and Feature Engineering".

Toutes les features sont calculees UNIQUEMENT a partir de la chaine URL
(features lexicales) : aucun acces reseau, aucune requete WHOIS, aucun
telechargement de contenu. C'est ce qui rend l'approche rapide et sans risque.
"""

import re
from urllib.parse import urlparse
import pandas as pd

# Mots "sensibles" frequemment presents dans les URLs de phishing
SUSPICIOUS_WORDS = [
    "secure", "account", "webscr", "login", "ebayisapi", "signin", "banking",
    "confirm", "hacker", "pirate", "paypal", "update", "bank", "hacking", "free",
]

# Services de raccourcissement d'URL connus
SHORTENERS = [
    "bit.ly", "goo.gl", "tinyurl", "ow.ly", "t.co", "is.gd", "buff.ly",
    "adf.ly", "j.mp", "cutt.ly", "shorte.st", "bc.vc", "tny.im",
]

IP_REGEX = re.compile(
    r"(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)"
)


def _ensure_scheme(url: str) -> str:
    """urlparse a besoin d'un schema pour isoler correctement le hostname."""
    if "://" not in url:
        return "http://" + url
    return url


def extract_features(url: str) -> dict:
    """Retourne un dictionnaire des 33 features pour une URL donnee."""
    url = str(url).strip()
    try:
        parsed = urlparse(_ensure_scheme(url))
        hostname = parsed.netloc or ""
        path = parsed.path or ""
        scheme = parsed.scheme
    except ValueError:
        # URL malformee (ex. crochets non valides) : fallback lexical simple
        no_scheme = url.split("://", 1)[-1]
        hostname = no_scheme.split("/", 1)[0]
        path = "/" + no_scheme.split("/", 1)[1] if "/" in no_scheme else ""
        scheme = url.split("://", 1)[0] if "://" in url else "http"

    # Premier repertoire du chemin
    dirs = [d for d in path.split("/") if d]
    first_dir = dirs[0] if dirs else ""

    # Top-Level Domain (approximation lexicale : dernier segment du hostname)
    host_no_port = hostname.split(":")[0]
    tld = host_no_port.split(".")[-1] if "." in host_no_port else ""

    low = url.lower()
    special_chars = re.findall(r"[&\-_~#%=?@.]", url)

    f = {
        # 1-3 : comptages de caracteres
        "length_url": len(url),
        "number_letter": sum(c.isalpha() for c in url),
        "numeric_character": sum(c.isdigit() for c in url),
        # 4-6 : extensions de domaine
        "number_dot_com": low.count(".com"),
        "number_dot_co": low.count(".co"),
        "number_dot_net": low.count(".net"),
        # 7-9
        "number_of_slash": url.count("/"),
        "number_of_uppercase": sum(c.isupper() for c in url),
        "number_of_lowercase": sum(c.islower() for c in url),
        # 10-12
        "number_dot_info": low.count(".info"),
        "number_of_https": low.count("https"),
        "number_of_http": low.count("http"),
        # 13-17
        "num_of_www_point": low.count("www."),
        "special_characters": len(special_chars),
        "num_of_percentage": url.count("%"),
        "question_mark": url.count("?"),
        "number_of_dash": url.count("-"),
        # 18 : dash dans le domaine (booleen)
        "dash_in_domain": int("-" in host_no_port),
        # 19
        "num_equal_symbol": url.count("="),
        # 20 : ratio de "%"
        "percentage_character": (url.count("%") / len(url)) if len(url) else 0.0,
        # 21 : espace encode dans le chemin
        "space_in_url": int("%20" in low),
        # 22 : mots suspects
        "suspicious_words": int(any(w in low for w in SUSPICIOUS_WORDS)),
        # 23 : IP a la place du domaine
        "ip_adress": int(bool(IP_REGEX.search(host_no_port))),
        # 24 : protocole https
        "protocol_https": int(scheme == "https"),
        # 25 : terme "https" present dans le hostname (technique de tromperie)
        "https_domaine": int("https" in host_no_port.lower()),
        # 26 : nombre de sous-domaines
        "number_sub_domain": max(host_no_port.count(".") - 1, 0),
        # 27 : lien raccourci
        "short_link": int(any(s in low for s in SHORTENERS)),
        # 28 : nombre de "@"
        "number_at": url.count("@"),
        # 29 : double slash de redirection (hors "://")
        "num_double_slash": max(url.count("//") - 1, 0),
        # 30-33 : longueurs
        "hostname_length": len(hostname),
        "path_length": len(path),
        "first_directory_length": len(first_dir),
        "tld_length": len(tld),
    }
    return f


def build_feature_dataframe(urls) -> pd.DataFrame:
    """Applique extract_features a une serie/liste d'URLs -> DataFrame de features."""
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows)


# Liste ordonnee des noms de features (utile pour verification / documentation)
FEATURE_NAMES = list(extract_features("http://example.com/path").keys())

if __name__ == "__main__":
    test = "http://secure-paypal.com.verify@192.168.0.1//login.php?account=1"
    import json
    print(f"Nombre de features : {len(FEATURE_NAMES)}")
    print(json.dumps(extract_features(test), indent=2))
