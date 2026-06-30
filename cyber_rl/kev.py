import datetime as _dt
import json
import random

from cyber_rl.env import CyberScenario, _add_edge, _empty_graph, _finalize_graph


HIGH_RISK_CWES = set([
    "CWE-20",
    "CWE-22",
    "CWE-78",
    "CWE-79",
    "CWE-89",
    "CWE-94",
    "CWE-119",
    "CWE-287",
    "CWE-306",
    "CWE-352",
    "CWE-434",
    "CWE-502",
    "CWE-787",
    "CWE-798",
])

HIGH_RISK_TERMS = [
    "remote code execution",
    "execute arbitrary code",
    "command injection",
    "authentication bypass",
    "privilege escalation",
    "deserialization",
    "path traversal",
    "sql injection",
    "zero-day",
]


def load_kev_catalog(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_year(cve_id):
    parts = cve_id.split("-")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return 0
    return 0


def parse_date(value):
    try:
        return _dt.date.fromisoformat(value[:10])
    except Exception:
        return _dt.date(1970, 1, 1)


def normalize_record(record):
    cwes = record.get("cwes") or []
    if isinstance(cwes, str):
        cwes = [cwes]
    return {
        "cveID": record.get("cveID", ""),
        "vendorProject": record.get("vendorProject", "Unknown"),
        "product": record.get("product", "Unknown"),
        "vulnerabilityName": record.get("vulnerabilityName", ""),
        "shortDescription": record.get("shortDescription", ""),
        "requiredAction": record.get("requiredAction", ""),
        "dateAdded": record.get("dateAdded", ""),
        "dueDate": record.get("dueDate", ""),
        "knownRansomwareCampaignUse": record.get("knownRansomwareCampaignUse", "Unknown"),
        "notes": record.get("notes", ""),
        "cwes": list(cwes),
        "year": parse_year(record.get("cveID", "")),
    }


def normalize_catalog(catalog):
    records = [normalize_record(record) for record in catalog.get("vulnerabilities", [])]
    return {
        "title": catalog.get("title"),
        "catalogVersion": catalog.get("catalogVersion"),
        "dateReleased": catalog.get("dateReleased"),
        "count": catalog.get("count", len(records)),
        "records": records,
    }


def risk_score(record):
    score = 1
    text = " ".join([
        record.get("vulnerabilityName", ""),
        record.get("shortDescription", ""),
    ]).lower()
    for term in HIGH_RISK_TERMS:
        if term in text:
            score += 1
    if set(record.get("cwes", [])) & HIGH_RISK_CWES:
        score += 1
    if record.get("knownRansomwareCampaignUse") == "Known":
        score += 1
    if record.get("year", 0) >= 2023:
        score += 1
    return max(1, min(5, score))


def node_value(record):
    value = 2.0 + float(risk_score(record))
    if record.get("knownRansomwareCampaignUse") == "Known":
        value += 2.0
    if record.get("year", 0) >= 2023:
        value += 1.0
    return min(value, 10.0)


def _select_records(records, family, rng, n_nodes):
    pool = list(records)
    if family == "ransomware_focus":
        known = [record for record in pool if record.get("knownRansomwareCampaignUse") == "Known"]
        pool = known if len(known) >= n_nodes else pool
        pool.sort(key=lambda record: (risk_score(record), record.get("year", 0)), reverse=True)
        start = rng.randrange(max(1, len(pool) - n_nodes + 1))
        return pool[start:start + n_nodes]
    if family == "recent_enterprise":
        pool.sort(key=lambda record: (parse_date(record.get("dateAdded", "")), risk_score(record)), reverse=True)
        top = pool[:max(n_nodes * 12, n_nodes)]
        return rng.sample(top, n_nodes)
    if family == "vendor_cluster":
        by_vendor = {}
        for record in pool:
            by_vendor.setdefault(record.get("vendorProject", "Unknown"), []).append(record)
        vendors = [vendor for vendor, values in by_vendor.items() if len(values) >= max(3, n_nodes // 2)]
        vendor = rng.choice(vendors) if vendors else rng.choice(list(by_vendor.keys()))
        selected = list(by_vendor[vendor])
        rng.shuffle(selected)
        if len(selected) < n_nodes:
            remainder = [record for record in pool if record not in selected]
            rng.shuffle(remainder)
            selected.extend(remainder)
        return selected[:n_nodes]
    if family == "cwe_cluster":
        by_cwe = {}
        for record in pool:
            for cwe in record.get("cwes", []):
                by_cwe.setdefault(cwe, []).append(record)
        cwes = [cwe for cwe, values in by_cwe.items() if len(values) >= max(3, n_nodes // 2)]
        cwe = rng.choice(cwes) if cwes else rng.choice(list(by_cwe.keys()))
        selected = list(by_cwe[cwe])
        rng.shuffle(selected)
        if len(selected) < n_nodes:
            remainder = [record for record in pool if record not in selected]
            rng.shuffle(remainder)
            selected.extend(remainder)
        return selected[:n_nodes]
    if family == "mixed_kev":
        return rng.sample(pool, n_nodes)
    raise ValueError("Unknown KEV scenario family: {}".format(family))


def _build_topology(family, selected, rng):
    n_nodes = len(selected)
    graph = _empty_graph(n_nodes)
    if family in ("recent_enterprise", "ransomware_focus"):
        for idx in range(n_nodes - 1):
            _add_edge(graph, idx, idx + 1)
        for idx in range(2, n_nodes):
            if rng.random() < 0.35:
                _add_edge(graph, rng.randrange(0, idx), idx)
    elif family == "vendor_cluster":
        for idx in range(1, n_nodes):
            _add_edge(graph, 0 if idx <= 2 else rng.randrange(1, idx), idx)
        for _ in range(n_nodes // 2):
            _add_edge(graph, rng.randrange(n_nodes), rng.randrange(n_nodes))
    elif family == "cwe_cluster":
        for idx in range(n_nodes - 1):
            _add_edge(graph, idx, idx + 1)
        for idx in range(0, n_nodes - 2, 2):
            _add_edge(graph, idx, idx + 2)
    elif family == "mixed_kev":
        for idx in range(1, n_nodes):
            _add_edge(graph, idx, rng.randrange(idx))
        for _ in range(n_nodes):
            _add_edge(graph, rng.randrange(n_nodes), rng.randrange(n_nodes))
    return _finalize_graph(graph)


def make_kev_scenario(catalog, family, seed, n_nodes=8, max_steps=24):
    normalized = normalize_catalog(catalog)
    records = normalized["records"]
    if len(records) < n_nodes:
        raise ValueError("KEV catalog has fewer records than requested nodes")
    rng = random.Random(seed)
    selected = _select_records(records, family, rng, n_nodes)
    selected = sorted(selected, key=lambda record: (risk_score(record), node_value(record)), reverse=True)

    # Keep the entry node low risk and target high risk so the path has a meaningful objective.
    target_record = selected[0]
    entry_record = selected[-1]
    middle = selected[1:-1]
    rng.shuffle(middle)
    ordered = [entry_record] + middle + [target_record]

    vulnerabilities = [risk_score(record) for record in ordered]
    values = [node_value(record) for record in ordered]
    values[-1] = 10.0
    adjacency = _build_topology(family, ordered, rng)

    node_metadata = []
    for idx, record in enumerate(ordered):
        node_metadata.append({
            "node": idx,
            "cveID": record["cveID"],
            "vendorProject": record["vendorProject"],
            "product": record["product"],
            "vulnerabilityName": record["vulnerabilityName"],
            "dateAdded": record["dateAdded"],
            "dueDate": record["dueDate"],
            "knownRansomwareCampaignUse": record["knownRansomwareCampaignUse"],
            "cwes": record["cwes"],
            "risk_score": vulnerabilities[idx],
            "value": values[idx],
        })

    return CyberScenario(
        name="kev_{}_seed{}".format(family, seed),
        adjacency=adjacency,
        entry_node=0,
        target_node=n_nodes - 1,
        vulnerabilities=vulnerabilities,
        values=values,
        max_steps=max_steps,
        detection_limit=6.0,
        patch_budget=3,
        decoy_budget=2,
        node_metadata=node_metadata,
        source={
            "dataset": "CISA Known Exploited Vulnerabilities Catalog",
            "catalogVersion": normalized["catalogVersion"],
            "dateReleased": normalized["dateReleased"],
            "count": normalized["count"],
            "family": family,
        },
    )


def kev_families():
    return ["recent_enterprise", "ransomware_focus", "vendor_cluster", "cwe_cluster", "mixed_kev"]
