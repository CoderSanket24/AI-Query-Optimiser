"""Deterministic, parameterized IMDb join-order evaluation workload."""

from __future__ import annotations

TEMPLATES = [
    {"id": "T01", "split": "train", "tables": ("t", "ci"), "params": (1980, 1985, 1990, 1995), "filters": ("title.production_year",), "sql": "SELECT MIN(t.title) FROM title t JOIN cast_info ci ON t.id=ci.movie_id WHERE t.production_year=%s"},
    {"id": "T02", "split": "train", "tables": ("t", "mk"), "params": (1980, 1985, 1990, 1995), "filters": ("title.production_year",), "sql": "SELECT MIN(t.title) FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id WHERE t.production_year=%s"},
    {"id": "T03", "split": "train", "tables": ("t", "mc"), "params": ((1980, 1), (1985, 2), (1990, 1), (1995, 2)), "filters": ("title.production_year", "movie_companies.company_type_id"), "sql": "SELECT MIN(t.title) FROM title t JOIN movie_companies mc ON t.id=mc.movie_id WHERE t.production_year=%s AND mc.company_type_id=%s"},
    {"id": "T04", "split": "train", "tables": ("t", "ci", "n"), "params": (1980, 1985, 1990, 1995), "filters": ("title.production_year",), "sql": "SELECT MIN(t.title) FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.production_year=%s"},
    {"id": "T05", "split": "train", "tables": ("t", "mk", "k"), "params": (1980, 1985, 1990, 1995), "filters": ("title.production_year",), "sql": "SELECT MIN(t.title) FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id JOIN keyword k ON mk.keyword_id=k.id WHERE t.production_year=%s"},
    {"id": "T06", "split": "train", "tables": ("t", "ci", "mk", "k"), "params": ((1980, 1), (1985, 2), (1990, 1), (1995, 2)), "filters": ("title.production_year", "title.kind_id"), "sql": "SELECT MIN(t.title) FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN movie_keyword mk ON t.id=mk.movie_id JOIN keyword k ON mk.keyword_id=k.id WHERE t.production_year=%s AND t.kind_id=%s"},
    {"id": "T07", "split": "test", "reuse_of": "T04", "tables": ("t", "ci", "n"), "params": (2000, 2001, 2005, 2015), "filters": ("title.production_year",), "sql": "SELECT MIN(t.title) FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.production_year=%s"},
    {"id": "T08", "split": "test", "reuse_of": "T05", "tables": ("t", "mk", "k"), "params": (2000, 2005, 2010, 2015), "filters": ("title.production_year",), "sql": "SELECT MIN(t.title) FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id JOIN keyword k ON mk.keyword_id=k.id WHERE t.production_year=%s"},
    {"id": "T09", "split": "test", "reuse_of": "T06", "tables": ("t", "ci", "mk", "k"), "params": ((2000, 1), (2005, 2), (2010, 3), (2015, 4)), "filters": ("title.production_year", "title.kind_id"), "sql": "SELECT MIN(t.title) FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN movie_keyword mk ON t.id=mk.movie_id JOIN keyword k ON mk.keyword_id=k.id WHERE t.production_year=%s AND t.kind_id=%s"},
    {"id": "T10", "split": "test", "test_only": True, "tables": ("ci", "n"), "params": ((1, 1000000), (1000000, 2000000), (2000000, 3000000), (3000000, 4000000)), "filters": ("cast_info.id",), "sql": "SELECT MIN(n.name) FROM cast_info ci JOIN name n ON ci.person_id=n.id WHERE ci.id >= %s AND ci.id < %s"},
    {"id": "T11", "split": "test", "test_only": True, "tables": ("mk", "k"), "params": ((1, 250000), (250000, 500000), (500000, 750000), (750000, 1000000)), "filters": ("movie_keyword.id",), "sql": "SELECT MIN(k.keyword) FROM movie_keyword mk JOIN keyword k ON mk.keyword_id=k.id WHERE mk.id >= %s AND mk.id < %s"},
    {"id": "T12", "split": "test", "test_only": True, "tables": ("mc", "cn"), "params": ((1, 250000), (250000, 500000), (500000, 750000), (750000, 1000000)), "filters": ("movie_companies.id",), "sql": "SELECT MIN(cn.name) FROM movie_companies mc JOIN company_name cn ON mc.company_id=cn.id WHERE mc.id >= %s AND mc.id < %s"},
]

EXTRA_UNSEEN = [
    {"id": "T13", "split": "test", "test_only": True, "tables": ("t", "mc", "cn"), "params": ((2000, 1), (2005, 2), (2010, 3), (2015, 4)), "filters": ("title.production_year", "movie_companies.company_type_id"), "sql": "SELECT MIN(cn.name) FROM title t JOIN movie_companies mc ON t.id=mc.movie_id JOIN company_name cn ON mc.company_id=cn.id WHERE t.production_year=%s AND mc.company_type_id=%s"},
    {"id": "T14", "split": "test", "test_only": True, "tables": ("t", "mk", "mc", "cn"), "params": ((2000, 1), (2005, 2), (2010, 3), (2015, 4)), "filters": ("title.production_year", "movie_companies.company_type_id"), "sql": "SELECT MIN(cn.name) FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id JOIN movie_companies mc ON t.id=mc.movie_id JOIN company_name cn ON mc.company_id=cn.id WHERE t.production_year=%s AND mc.company_type_id=%s"},
]

ALIAS_TO_TABLE = {"t": "title", "ci": "cast_info", "n": "name", "mk": "movie_keyword", "k": "keyword", "mc": "movie_companies", "cn": "company_name"}


def get_templates(extra_unseen=False):
    return TEMPLATES + (EXTRA_UNSEEN if extra_unseen else [])


def build_queries(connection, extra_unseen=False):
    """Return query instances after verifying every filter column in information_schema."""
    templates = get_templates(extra_unseen)
    required = sorted({tuple(name.split(".", 1)) for template in templates for name in template["filters"]})
    with connection.cursor() as cur:
        cur.execute("SELECT table_name, column_name FROM information_schema.columns WHERE table_schema='public'")
        present = set(cur.fetchall())
    missing = [f"{table}.{column}" for table, column in required if (table, column) not in present]
    if missing:
        raise RuntimeError("Workload filter columns missing from information_schema: " + ", ".join(missing))

    queries = []
    for template in templates:
        for index, param_values in enumerate(template["params"], 1):
            params = param_values if isinstance(param_values, tuple) else (param_values,)
            filter_sql, filter_params = {}, {}
            param_index = 0
            if "title.production_year" in template["filters"]:
                filter_sql["t"] = " WHERE production_year=%s"
                filter_params["t"] = (params[param_index],)
                param_index += 1
                if "title.kind_id" in template["filters"]:
                    filter_sql["t"] += " AND kind_id=%s"
                    filter_params["t"] += (params[param_index],)
                    param_index += 1
            if "movie_companies.company_type_id" in template["filters"]:
                filter_sql["mc"] = " WHERE company_type_id=%s"
                filter_params["mc"] = (params[param_index],)
                param_index += 1
            for table_name, alias, column in (
                ("cast_info", "ci", "id"),
                ("movie_keyword", "mk", "id"),
                ("movie_companies", "mc", "id"),
            ):
                if f"{table_name}.{column}" in template["filters"]:
                    filter_sql[alias] = f" WHERE {column} >= %s AND {column} < %s"
                    filter_params[alias] = params
            queries.append({
                "query_id": f"{template['id']}-{index}",
                "template_id": template["id"],
                "split": template["split"],
                "reuse_of": template.get("reuse_of", ""),
                "test_only": template.get("test_only", False),
                "tables": template["tables"],
                "table_set": "{" + ",".join(template["tables"]) + "}",
                "n_tables": len(template["tables"]),
                "filters": template["filters"],
                "params": params,
                "filter_sql": filter_sql,
                "filter_params": filter_params,
                "filter_aliases": tuple(filter_sql),
                "sql": template["sql"],
            })
    return queries
