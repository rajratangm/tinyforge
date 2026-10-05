from tinyforge.ft_task import sql_extract, sql_norm, sql_score, sql_valid

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
