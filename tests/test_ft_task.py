from tinyforge.ft_task import sql_exec_match, sql_extract, sql_norm, sql_score, sql_valid, wilson_ci

SCHEMA = "CREATE TABLE head (age INTEGER, name TEXT)"
USER = f"How many heads are older than 56?\n\n{SCHEMA}"


def test_extract_handles_fences_and_chatter():
    assert sql_extract("```sql\nSELECT 1;\n```\nThis query...") == "SELECT 1;"
    assert sql_extract("SELECT 1\n\nExplanation: blah") == "SELECT 1"


def test_norm_ignores_case_spacing_semicolon_quotes():
    a = 'SELECT COUNT(*) FROM head WHERE age > 56;'
    b = "select  count( * )  from head where age>56"
    assert sql_norm(a) == sql_norm(b)
    assert sql_norm('SELECT "name" FROM head') == sql_norm("select 'name' from head")


def test_validity_uses_schema():
    assert sql_valid("SELECT COUNT(*) FROM head WHERE age > 56", SCHEMA)
    assert not sql_valid("SELECT COUNT(*) FROM nope", SCHEMA)
    assert not sql_valid("SELEKT broken", SCHEMA)


def test_score_separates_strict_from_lenient():
    ref = "SELECT COUNT(*) FROM head WHERE age > 56"
    chatty = f"Sure! Here is the query:\n\n```sql\n{ref}\n```"
    s = sql_score(USER, chatty, ref)
    assert s["lenient_em"] and not s["strict_em"] and s["valid"]
    assert sql_score(USER, ref, ref)["strict_em"]
    wrong = sql_score(USER, "SELECT name FROM head", ref)
    assert not wrong["lenient_em"] and wrong["valid"]


def test_exec_match_accepts_different_text_same_result():
    ref = "SELECT COUNT(*) FROM head WHERE age > 56"
    assert sql_exec_match("select count(name) from head where 56 < age", ref, SCHEMA)


def test_exec_match_rejects_wrong_result_and_errors():
    ref = "SELECT COUNT(*) FROM head WHERE age > 56"
    assert not sql_exec_match("SELECT COUNT(*) FROM head WHERE age > 10", ref, SCHEMA)
    assert not sql_exec_match("SELECT COUNT(*) FROM nope", ref, SCHEMA)
    assert not sql_exec_match("SELEKT", ref, SCHEMA)


def test_exec_match_respects_order_only_when_gold_orders():
    unordered = "SELECT name FROM head WHERE age > 10"
    assert sql_exec_match("SELECT name FROM head WHERE age > 10 ORDER BY name DESC", unordered, SCHEMA)
    ordered = "SELECT name FROM head ORDER BY name ASC"
    assert not sql_exec_match("SELECT name FROM head ORDER BY name DESC", ordered, SCHEMA)


def test_exec_match_runaway_query_is_stopped():
    big = "SELECT COUNT(*) FROM head a, head b, head c, head d, head e, head f, head g, head h"
    ref = "SELECT COUNT(*) FROM head"
    assert not sql_exec_match(big, ref, SCHEMA)


def test_exec_match_no_evidence_is_not_a_match():
    # gold returns no rows on any generated database (impossible predicate) -> cannot claim a match
    assert not sql_exec_match("SELECT name FROM head WHERE 1=0", "SELECT name FROM head WHERE 1=0", SCHEMA)


def test_score_reports_exec_acc():
    ref = "SELECT COUNT(*) FROM head WHERE age > 56"
    assert sql_score(USER, "SELECT COUNT(name) FROM head WHERE age > 56", ref)["exec_acc"]


def test_wilson_ci_properties():
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi and 0.40 < lo and hi < 0.60
    assert wilson_ci(0, 0) == [0.0, 1.0]
    assert wilson_ci(10, 10)[1] == 1.0 and wilson_ci(10, 10)[0] > 0.6


def test_exec_match_column_order_matters():
    schema = "CREATE TABLE t (a INTEGER, b INTEGER)"
    assert not sql_exec_match("SELECT b, a FROM t", "SELECT a, b FROM t", schema)


def test_literal_regex_extracts_text_and_numbers():
    from tinyforge.ft_task import _literals
    assert _literals("SELECT x FROM t WHERE n = 'april 6' AND k > 56") == (["april 6"], [56.0])
