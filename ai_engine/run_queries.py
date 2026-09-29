"""
run_queries.py
--------------
Training script for the PPO RL agent.
Fires 20 diverse JOIN queries through Spring Boot so the agent
learns which table order produces the fastest execution time.

Run multiple times to accumulate training experience:
  venv\Scripts\python run_queries.py

After 5+ runs (100 queries), check attention weights at:
  GET http://localhost:8000/feedback/stats
"""

import urllib.request
import json
import time

# Spring Boot middleware endpoint
URL     = "http://localhost:8080/api/query/execute"
HEADERS = {"Content-Type": "application/json"}

# 20 diverse queries mixing 2-table, 3-table, and 4-table joins
QUERIES = [
    # 3-Table: title + cast_info + name
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.production_year=1990 LIMIT 10",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.production_year=2000 LIMIT 10",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.production_year=2010 LIMIT 10",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.kind_id=1 LIMIT 10",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.kind_id=2 LIMIT 10",

    # 3-Table: title + movie_keyword + keyword
    "SELECT t.title, k.keyword FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id JOIN keyword k ON mk.keyword_id=k.id WHERE t.production_year=1995 LIMIT 10",
    "SELECT t.title, k.keyword FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id JOIN keyword k ON mk.keyword_id=k.id WHERE t.production_year=2005 LIMIT 10",
    "SELECT t.title, k.keyword FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id JOIN keyword k ON mk.keyword_id=k.id WHERE t.production_year=2010 LIMIT 10",

    # 3-Table: title + movie_companies + company_name
    "SELECT t.title, cn.name FROM title t JOIN movie_companies mc ON t.id=mc.movie_id JOIN company_name cn ON mc.company_id=cn.id WHERE t.production_year=2000 LIMIT 10",
    "SELECT t.title, cn.name FROM title t JOIN movie_companies mc ON t.id=mc.movie_id JOIN company_name cn ON mc.company_id=cn.id WHERE t.production_year=2010 LIMIT 10",

    # 2-Table (simple baselines)
    "SELECT t.title FROM title t JOIN cast_info ci ON t.id=ci.movie_id WHERE t.production_year=1998 LIMIT 10",
    "SELECT t.title FROM title t JOIN cast_info ci ON t.id=ci.movie_id WHERE t.kind_id=1 LIMIT 10",
    "SELECT t.title FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id WHERE t.production_year=2002 LIMIT 10",
    "SELECT t.title FROM title t JOIN movie_companies mc ON t.id=mc.movie_id WHERE t.production_year=2012 LIMIT 10",

    # 4-Table (heavy joins)
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id JOIN movie_keyword mk ON t.id=mk.movie_id WHERE t.production_year=1995 LIMIT 5",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id JOIN movie_keyword mk ON t.id=mk.movie_id WHERE t.production_year=2008 LIMIT 5",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id JOIN movie_companies mc ON t.id=mc.movie_id WHERE t.production_year=1999 LIMIT 5",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id JOIN movie_companies mc ON t.id=mc.movie_id WHERE t.production_year=2007 LIMIT 5",

    # More 3-table variations
    "SELECT t.title, k.keyword FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id JOIN keyword k ON mk.keyword_id=k.id WHERE t.production_year=2015 LIMIT 10",
    "SELECT t.title, cn.name FROM title t JOIN movie_companies mc ON t.id=mc.movie_id JOIN company_name cn ON mc.company_id=cn.id WHERE t.production_year=1990 LIMIT 10",
]


def run():
    print(f"Firing {len(QUERIES)} queries through Spring Boot -> FastAPI -> PostgreSQL...\n")
    success = 0
    failed  = 0

    for i, sql in enumerate(QUERIES, 1):
        payload = json.dumps({"query": sql}).encode("utf-8")
        req     = urllib.request.Request(URL, data=payload, headers=HEADERS, method="POST")

        print(f"[{i:02d}/{len(QUERIES)}] Running...")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                res       = json.loads(resp.read())
                exec_time = res.get("exec_time_ms", 0)
                order     = " -> ".join(res.get("choosen_order", []))
                print(f"         Order : {order}")
                print(f"         Exec  : {exec_time} ms\n")
                success += 1
        except Exception as e:
            print(f"         ERROR : {e}\n")
            failed += 1

        time.sleep(0.5)   # small delay so terminal is readable

    print("=" * 50)
    print(f"Done. Success: {success} | Failed: {failed}")
    print("Check training progress at: GET http://localhost:8000/feedback/stats")
    print("=" * 50)


if __name__ == "__main__":
    run()
