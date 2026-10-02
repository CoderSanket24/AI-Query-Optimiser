import urllib.request
import json
import time

URL     = "http://localhost:8080/api/query/execute"
HEADERS = {"Content-Type": "application/json"}

QUERIES = [
    "SELECT t.title FROM title t JOIN cast_info ci ON t.id=ci.movie_id WHERE t.production_year=1980 LIMIT 10",
    "SELECT t.title FROM title t JOIN cast_info ci ON t.id=ci.movie_id WHERE t.production_year=1985 LIMIT 10",
    "SELECT t.title FROM title t JOIN cast_info ci ON t.id=ci.movie_id WHERE t.production_year=1990 LIMIT 10",
    "SELECT t.title FROM title t JOIN cast_info ci ON t.id=ci.movie_id WHERE t.production_year=1995 LIMIT 10",
    "SELECT t.title FROM title t JOIN cast_info ci ON t.id=ci.movie_id WHERE t.production_year=2000 LIMIT 10",
    "SELECT t.title FROM title t JOIN cast_info ci ON t.id=ci.movie_id WHERE t.production_year=2005 LIMIT 10",
    "SELECT t.title FROM title t JOIN cast_info ci ON t.id=ci.movie_id WHERE t.production_year=2010 LIMIT 10",
    "SELECT t.title FROM title t JOIN cast_info ci ON t.id=ci.movie_id WHERE t.kind_id=1 LIMIT 10",
    "SELECT t.title FROM title t JOIN cast_info ci ON t.id=ci.movie_id WHERE t.kind_id=3 LIMIT 10",

    "SELECT t.title FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id WHERE t.production_year=1992 LIMIT 10",
    "SELECT t.title FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id WHERE t.production_year=2003 LIMIT 10",
    "SELECT t.title FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id WHERE t.production_year=2011 LIMIT 10",

    "SELECT t.title FROM title t JOIN movie_companies mc ON t.id=mc.movie_id WHERE t.production_year=1998 LIMIT 10",
    "SELECT t.title FROM title t JOIN movie_companies mc ON t.id=mc.movie_id WHERE t.production_year=2007 LIMIT 10",
    "SELECT t.title FROM title t JOIN movie_companies mc ON t.id=mc.movie_id WHERE mc.company_type_id=1 LIMIT 10",
    "SELECT t.title FROM title t JOIN movie_companies mc ON t.id=mc.movie_id WHERE mc.company_type_id=2 LIMIT 10",

    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.production_year=1975 LIMIT 10",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.production_year=1982 LIMIT 10",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.production_year=1990 LIMIT 10",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.production_year=1997 LIMIT 10",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.production_year=2002 LIMIT 10",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.production_year=2008 LIMIT 10",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.production_year=2013 LIMIT 10",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.kind_id=1 LIMIT 10",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.kind_id=2 LIMIT 10",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.kind_id=4 LIMIT 10",

    "SELECT t.title, k.keyword FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id JOIN keyword k ON mk.keyword_id=k.id WHERE t.production_year=1988 LIMIT 10",
    "SELECT t.title, k.keyword FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id JOIN keyword k ON mk.keyword_id=k.id WHERE t.production_year=1994 LIMIT 10",
    "SELECT t.title, k.keyword FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id JOIN keyword k ON mk.keyword_id=k.id WHERE t.production_year=2001 LIMIT 10",
    "SELECT t.title, k.keyword FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id JOIN keyword k ON mk.keyword_id=k.id WHERE t.production_year=2006 LIMIT 10",
    "SELECT t.title, k.keyword FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id JOIN keyword k ON mk.keyword_id=k.id WHERE t.production_year=2012 LIMIT 10",
    "SELECT t.title, k.keyword FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id JOIN keyword k ON mk.keyword_id=k.id WHERE t.production_year=2015 LIMIT 10",

    "SELECT t.title, cn.name FROM title t JOIN movie_companies mc ON t.id=mc.movie_id JOIN company_name cn ON mc.company_id=cn.id WHERE t.production_year=1985 LIMIT 10",
    "SELECT t.title, cn.name FROM title t JOIN movie_companies mc ON t.id=mc.movie_id JOIN company_name cn ON mc.company_id=cn.id WHERE t.production_year=1993 LIMIT 10",
    "SELECT t.title, cn.name FROM title t JOIN movie_companies mc ON t.id=mc.movie_id JOIN company_name cn ON mc.company_id=cn.id WHERE t.production_year=2000 LIMIT 10",
    "SELECT t.title, cn.name FROM title t JOIN movie_companies mc ON t.id=mc.movie_id JOIN company_name cn ON mc.company_id=cn.id WHERE t.production_year=2009 LIMIT 10",
    "SELECT t.title, cn.name FROM title t JOIN movie_companies mc ON t.id=mc.movie_id JOIN company_name cn ON mc.company_id=cn.id WHERE t.production_year=2014 LIMIT 10",

    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id JOIN movie_keyword mk ON t.id=mk.movie_id WHERE t.production_year=1990 LIMIT 5",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id JOIN movie_keyword mk ON t.id=mk.movie_id WHERE t.production_year=1995 LIMIT 5",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id JOIN movie_keyword mk ON t.id=mk.movie_id WHERE t.production_year=2000 LIMIT 5",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id JOIN movie_keyword mk ON t.id=mk.movie_id WHERE t.production_year=2005 LIMIT 5",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id JOIN movie_keyword mk ON t.id=mk.movie_id WHERE t.production_year=2010 LIMIT 5",

    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id JOIN movie_companies mc ON t.id=mc.movie_id WHERE t.production_year=1992 LIMIT 5",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id JOIN movie_companies mc ON t.id=mc.movie_id WHERE t.production_year=1999 LIMIT 5",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id JOIN movie_companies mc ON t.id=mc.movie_id WHERE t.production_year=2004 LIMIT 5",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id JOIN movie_companies mc ON t.id=mc.movie_id WHERE t.production_year=2008 LIMIT 5",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id JOIN movie_companies mc ON t.id=mc.movie_id WHERE t.production_year=2012 LIMIT 5",

    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.production_year BETWEEN 1990 AND 1995 LIMIT 10",
    "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.production_year BETWEEN 2000 AND 2005 LIMIT 10",
]


def run():
    total = len(QUERIES)
    print(f"Firing {total} queries through Spring Boot -> FastAPI -> PostgreSQL...\n")
    success = 0
    failed  = 0

    for i, sql in enumerate(QUERIES, 1):
        payload = json.dumps({"query": sql}).encode("utf-8")
        req     = urllib.request.Request(URL, data=payload, headers=HEADERS, method="POST")

        print(f"[{i:02d}/{total}] Running...")
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

        time.sleep(0.5)

    print("=" * 50)
    print(f"Done. Success: {success} | Failed: {failed}")
    print("Check stats at: GET http://localhost:8000/feedback/stats")
    print("=" * 50)


if __name__ == "__main__":
    run()
