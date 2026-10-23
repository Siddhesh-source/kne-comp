"""Deterministic multilingual rule-based labeler for knee MRI reports.

Each report is split into clauses. A finding is *mentioned* in a clause when its
anatomy pattern and its lesion pattern both match (or a standalone pattern
matches). A mention is scored:

  1.0  asserted         ("complete tear of the ACL")
  0.5  hedged / mild    ("suspected tear", "grade 1 sprain")
  0.0  negated          ("no evidence of tear", "sin signos de rotura")

The study score for a label is the maximum over its mentions; silence is 0.
This plays the role of a hand-written hypothesis (inductive bias) that turns
free text into weak supervision for the image models.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from . import LABELS


def normalize(text: str) -> str:
    text = str(text).replace("µ", "μ")  # Greek reports use the micro sign for mu
    text = text.replace("ı", "i")  # Turkish dotless i
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def _rx(*parts: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in parts))


TEAR = _rx(
    r"\btear", r"\btorn\b", r"ruptur", r"\brupture", r"disruption", r"discontinu",
    r"bucket ?handle", r"\bdeficien", r"avuls",
    r"\brotura", r"desgarr", r"\broto\b", r"\brota\b",
    r"\briss\b", r"\brisse", r"\bruptur", r"abriss", r"\bausriss",
    r"\bscheur", r"ruptuur",
    r"dechirure", r"fissure meniscale",
    r"yirtik", r"rupturu", r"ruptur",
    r"\bruptura", r"\bpuknu",
    r"руптур", r"скъсван", r"разкъсван",
    r"ρηξη", r"ρηξεις",
)
INJURY = _rx(
    TEAR.pattern, r"\bsprain", r"\binjur", r"grade (?:2|3|ii|iii)\b", r"\blesion",
    r"esguince", r"\blesion", r"distorsi", r"\bletsel", r"\bverstuiking", r"\bentorse",
    r"\bzerrung", r"\bsprain", r"\bistegnu", r"\bозлед", r"\bлезия", r"\bκακωσ",
)
MENISCAL_TEAR = _rx(TEAR.pattern, r"\blasion", r"\blesion", r"meniscectom", r"meniskektom",
                    r"\bflap\b", r"\bmacerat", r"\bextru", r"\bkova sap", r"\bramp les")

NEG = _rx(
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative for\b", r"\babsence\b", r"\bnor\b",
    r"\bnone\b", r"\bfree of\b", r"\bsin\b", r"\bni\b", r"\bkein", r"\bohne\b", r"\bnicht\b",
    r"\bgeen\b", r"\bzonder\b", r"\bniet\b", r"\bpas de\b", r"\bsans\b", r"\bnao\b", r"\bsem\b",
    r"\byok", r"\bbez\b", r"\bnema\b", r"\bбез\b", r"\bняма\b", r"\bδεν\b", r"\bχωρις\b",
    r"\bexclud", r"\bdescart", r"\bausgeschlossen", r"\bnormal", r"\bintact", r"\bintakt",
)
NEG_AFTER = _rx(
    r"\bnot (?:seen|identified|present|demonstrated|visualized|torn)", r"\bunremarkable",
    r"\bizlenmedi", r"\bsaptanmadi", r"\byoktur\b", r"\bnormaldir", r"\bwithin normal",
    r"\bdentro de (?:los )?limites normales", r"\bsin alteraciones", r"\bconservad",
    r"\bintact", r"\bintakt", r"\bregelrecht", r"\bunauffallig", r"\bnormaal", r"\bnormal\b",
    r"\bodrzan", r"\buredn", r"\bprimjeren", r"\bнормал", r"\bинтакт", r"\bб\.о\.", r"\bφυσιολογ",
)
HEDGE = _rx(
    r"\bsuspect", r"\bpossib", r"\bprobabl", r"\br/o\b", r"\brule out", r"\bcannot be excluded",
    r"\bcan't (?:be )?(?:totally )?exclud", r"\bquestionable", r"\bequivocal", r"\bmay represent",
    r"\bsospech", r"\bposible", r"\bpodria", r"\bverdacht", r"\bfraglich", r"\bdd\b",
    r"\bmogelijk", r"\bsuspicion", r"\beventuel", r"\bsuphe", r"\bmoguc", r"\bvjerojatn",
    r"\bвъзможн", r"\bсъмнени", r"\bπιθαν", r"\bgrade (?:1|i)\b", r"\bgrado (?:1|i)\b",
    r"\bgraad (?:1|i)\b", r"\bin favou?r of\b", r"\bmight\b",
)
HISTORY = _rx(
    r"\bindication", r"\bclinical (?:history|information|indication)", r"\bhistory\b",
    r"\bantecedentes", r"\bklinische", r"\bvraagstel", r"\bfragestellung", r"\banamne",
    r"\bmotivo", r"\bcomparison",
)


@dataclass(frozen=True)
class Rule:
    anatomy: re.Pattern | None
    lesion: re.Pattern


MEDIAL_MENISCUS = (
    r"medial menisc|menisc\w* (?:interno|medial)|meniscus medialis|innenmenisk|\bim\b(?:-| )?(?:hinterhorn|vorderhorn|riss)?"
    r"|mediale menisc|menisque (?:interne|medial)|medial meniskus|medial meniskus|medial menisku"
    r"|medijaln\w* menisk|медиалн\w* менискус|εσω μηνισκ|μηνισκ\w* (?:στο )?εσω|both menisci|ambos meniscos"
    r"|beide[n]? menisken|medial and lateral menisc|medial & lateral menisc|oba meniska"
    r"|menisco interno y (?:lateral|externo)|medial ve lateral menisk"
)
LATERAL_MENISCUS = (
    r"lateral menisc|menisc\w* (?:externo|lateral)|meniscus lateralis|aussenmenisk|außenmenisk|\bam\b(?:-| )?(?:hinterhorn|vorderhorn|riss)?"
    r"|laterale menisc|menisque (?:externe|lateral)|lateral meniskus|lateral menisku"
    r"|lateraln\w* menisk|латерал\w* менискус|εξω μηνισκ|both menisci|ambos meniscos"
    r"|beide[n]? menisken|medial and lateral menisc|medial & lateral menisc|oba meniska"
    r"|(?:interno|medial) y (?:lateral|externo)|medial ve lateral menisk"
)
CARTILAGE = (
    r"chondr|cartilag|cartilago|knorpel|kraakbeen|cartilage|hrskavic|хрущял|χονδρ|kikirdak"
    r"|\boa\b|osteoarthr|arthros|artros|gonartr|artrotsk|artroz|артроз|οστεοαρθρ|osteophyt|osteofit|остеофит|οστεοφυτ"
    r"|ulcera condral|condropat|hondromalac|kondromalaz|chondromal|kraakbeenlijden"
)
MEDIAL_SIDE = (
    r"medial (?:femoral|tibial|compartment|tibiofemoral|femorotibial|condyl|plateau|joint)"
    r"|femorotibial medial|compartimento femorotibial medial|condilo femoral (?:medial|interno)"
    r"|platillo tibial (?:medial|interno)|mediale[n]? (?:femur|tibia|kompartiment|femorotib|condyl)"
    r"|mediaal femorotibia|mediale femorale|medial(?:e|en|es)? gelenkkompart|femorotibial(?:e[n]?)? medial"
    r"|medijaln\w* (?:kompartment|kondil|plato)|medijalnog kompartmenta|медиалн\w* (?:кондил|плато)"
    r"|εσω (?:διαμερισμα|μηριαι|κνημιαι|μεσαρθρι)|medial kondil|medial plato"
)
LATERAL_SIDE = (
    r"lateral (?:femoral|tibial|compartment|tibiofemoral|femorotibial|condyl|plateau|joint)"
    r"|femorotibial lateral|compartimento femorotibial lateral|condilo femoral (?:lateral|externo)"
    r"|platillo tibial (?:lateral|externo)|laterale[n]? (?:femur|tibia|kompartiment|femorotib|condyl)"
    r"|lateraal femorotibia|laterale femorale|lateral(?:e|en|es)? gelenkkompart|femorotibial(?:e[n]?)? lateral"
    r"|lateraln\w* (?:kompartment|kondil|plato)|латерал\w* (?:кондил|плато)"
    r"|εξω (?:διαμερισμα|μηριαι|κνημιαι)|lateral kondil|lateral plato"
)
PF_SIDE = (
    r"patell|patel|rotul|trochle|troclea|troklea|femoropatell|patelofemoral|patellofemoral"
    r"|kniescheibe|retropatell|\bpf\b|\bfp\b|пател|επιγονατιδ|trohlej"
)
TRICOMP = (
    r"tricompartment|three compartments|all three compartmen|tres compartimentos|gonartros|gonartrose"
    r"|panartros|tricompartimental|drei kompartiment"
)

RULES: dict[str, list[Rule]] = {
    "ACL": [Rule(_rx(r"\bacl\b", r"anterior cruciate", r"\blca\b", r"\bvkb\b", r"vordere[sn]? kreuzband",
                     r"voorste kruisband", r"ligamento cruzado anterior", r"ligament croise anterieur",
                     r"on capraz", r"prednj\w* kriz", r"предн\w* кръстн", r"προσθι\w* χιαστ", r"cruzado anterior"),
                 INJURY)],
    "MCL": [Rule(_rx(r"\bmcl\b", r"medial collateral", r"\blcm\b", r"\blli\b", r"colateral (?:medial|interno)",
                     r"lateral interno", r"innenband", r"mediale[sn]? (?:collateral|kollateral)", r"\bmcl\b",
                     r"medial kollateral", r"medijaln\w* kolateral", r"медиал\w* колатерал", r"εσω πλαγι"),
                 INJURY)],
    "Medial Meniscus": [Rule(re.compile(MEDIAL_MENISCUS), MENISCAL_TEAR)],
    "Lateral Meniscus": [Rule(re.compile(LATERAL_MENISCUS), MENISCAL_TEAR)],
    "Medial OA": [Rule(re.compile(MEDIAL_SIDE), re.compile(CARTILAGE)), Rule(None, re.compile(TRICOMP))],
    "Lateral OA": [Rule(re.compile(LATERAL_SIDE), re.compile(CARTILAGE)), Rule(None, re.compile(TRICOMP))],
    "PF OA": [Rule(re.compile(PF_SIDE), re.compile(CARTILAGE)), Rule(None, re.compile(TRICOMP))],
    "Effusion": [Rule(None, _rx(r"effusion", r"derrame", r"erguss", r"hydrops", r"epanchement", r"efuzyon",
                                r"izljev", r"излив", r"συλλογη υγρου", r"hemarthros", r"hemartros",
                                r"gewrichtsvocht", r"vocht in het gewricht", r"opzetting van (?:de )?suprapatellair",
                                r"sivi artis", r"distended suprapatellar", r"fluid accumulat"))],
    "Synovitis": [Rule(None, _rx(r"synovit", r"sinovit", r"synovialit", r"sinovitis", r"synovial (?:thicken|hypertroph|prolif)",
                                 r"thicken\w* (?:of the )?synovi", r"hypertrophy of the synovium", r"verdikking\w* van (?:het )?synovium",
                                 r"synovium.{0,20}verdik", r"proliferacij\w* sinovij", r"sinovij", r"синовит", r"υμενιτιδ",
                                 r"παχυνση του υμενα", r"engrosamiento sinovial", r"hoffitis"))],
    "Baker's": [Rule(None, _rx(r"baker", r"popliteal cyst", r"quiste[s]? popliteo", r"poplitea\w* cyst",
                               r"kyste poplite", r"бекер", r"poplitealna cist", r"poplitealn\w* cist",
                               r"cisto poplite", r"poplitealzyste", r"popliteale cyste", r"bekerova"))],
    "Contusion": [Rule(None, _rx(r"contusi", r"kontuzi", r"kontuzyon", r"контузи", r"bone bruise", r"bruise",
                                 r"botcontusie", r"\bμωλωπ", r"kissing",
                                 r"(?:bone marrow|osseous|bony|marrow) (?:edema|oedema)",
                                 r"edema oseo", r"knochenodem", r"knochenmarkodem", r"botoedeem",
                                 r"kemik iligi odem", r"kostani edem", r"koštani edem", r"костно-?мозъчен едем",
                                 r"οστεομυελικ\w* οιδημα"))],
    "Fracture": [Rule(None, _rx(r"fractur", r"fraktur", r"fractuur", r"frakture", r"kirik\w*", r"фрактур",
                                r"καταγμ", r"impaction", r"impactie", r"impresijsk", r"avulsion fracture",
                                r"segond", r"\bsonk\b", r"insufficiency fracture", r"depression of the lateral articular",
                                r"impression\w* fraktur", r"osteochondral impaction", r"fx\b"))],
}

SPLIT = re.compile(r"(?:[.;:!\n\r*>•]|\s-\s|\s--\s|\d\.\s|\[bilateral note[^\]]*\])+")


def clauses(report: str) -> list[str]:
    parts = [p.strip() for p in SPLIT.split(normalize(report))]
    return [p for p in parts if len(p) > 2]


def _mention_score(clause: str, start: int, end: int) -> float:
    before = clause[:start]
    window = " ".join(before.split()[-7:])
    if NEG.search(window):
        return 0.0
    after = clause[end:end + 40]
    if NEG_AFTER.search(after) and not TEAR.search(after):
        return 0.0
    context = " ".join(before.split()[-6:]) + " " + clause[start:end + 12]
    if HEDGE.search(context):
        return 0.5
    return 1.0


def _rule_score(rule: Rule, clause: str) -> float | None:
    if rule.anatomy is not None:
        anat = rule.anatomy.search(clause)
        if anat is None:
            return None
    best = None
    for match in rule.lesion.finditer(clause):
        start, end = match.span()
        if rule.anatomy is not None:
            # A negation between anatomy and lesion ("meniscus is not torn") negates it.
            if anat.end() <= start and NEG.search(clause[anat.end():start]):
                best = 0.0 if best is None else best
                continue
            # Anatomy decides negation when it precedes the lesion ("no ACL tear").
            start = min(start, anat.start())
            end = max(end, anat.end())
        score = _mention_score(clause, start, end)
        best = score if best is None else max(best, score)
    return best


def label_report(report: str) -> dict[str, float]:
    """Return {label: score in [0, 1]}; unmentioned findings score 0."""
    scores = {label: 0.0 for label in LABELS}
    for clause in clauses(report):
        if "?" in clause or HISTORY.search(clause[:40]):
            continue
        for label, rules in RULES.items():
            for rule in rules:
                score = _rule_score(rule, clause)
                if score is not None and score > scores[label]:
                    scores[label] = score
    return scores


def label_frame(reports) -> "pd.DataFrame":
    import pandas as pd

    return pd.DataFrame([label_report(r) for r in reports])
