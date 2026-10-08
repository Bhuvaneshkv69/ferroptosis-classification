"""Single source of truth for model names shown to users (app, reports). Stored artifacts keep their original metadata;
names are mapped here at display time so every surface uses the same scientific terms."""

CORE = "Ferroptosis Core Classifier"
MMR = "35-gene + MMR Extended Model"
EXTENDED = "Extended Molecular Model"
WT = "Whole-transcriptome model (experimental)"
DRUG = "Drug-response regression"

NAME = {"core": CORE, "core_plus_mmr": MMR, "extended_molecular": EXTENDED, "whole_transcriptome": WT}
WITH_GENES = {"core": f"{CORE} (35 genes)", "core_plus_mmr": f"{MMR} (39 genes)",
              "extended_molecular": f"{EXTENDED} (53 features)", "whole_transcriptome": WT}
FERROPTOSIS_ONLY = {"core": True, "core_plus_mmr": False, "extended_molecular": False, "whole_transcriptome": False}


def feature_set_of(entry: dict) -> str:
    """Feature-set key of a registry entry ('core', 'core_plus_mmr', 'extended_molecular'); legacy 35-gene entries are 'core'."""
    fp = str(entry.get("feature_pipeline", ""))
    head = fp.split("|")[0]
    return head if head in NAME else "core"


def display(entry: dict, with_genes: bool = True) -> str:
    fs = feature_set_of(entry)
    base = (WITH_GENES if with_genes else NAME)[fs]
    if str(entry.get("name", "")).startswith(("reference_", "auto_")):
        kind = "ensemble"
    else:                                   # stored names look like "<old feature-set label>: <algorithm>"; keep the algorithm only
        kind = str(entry.get("display_name", entry.get("name"))).split(": ")[-1]
    return f"{base}: {kind}"


def drug_display(entry: dict) -> str:
    """Drug-response model name: a single regressor (tree 'ensembles' inside one model are not a multi-model ensemble)."""
    return str(entry.get("display_name", entry.get("name"))).replace(" ensemble", " regressor") + " (single model)"
