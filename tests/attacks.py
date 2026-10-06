"""Attack corpus for the SQL safety stack.

Two buckets, because they make different promises:

WRITE_ATTACKS  each one tries to change the database. The design promise is that
               every one of the three layers stops it ON ITS OWN, so a bug in
               one layer cannot expose the data. Tests run each layer alone.
ABUSE_ATTACKS  not writes (side effects, schema probing, resource abuse, smuggled
               statements). The full pipeline must refuse them; which layer does
               so is not part of the contract.
"""

WRITE_ATTACKS = [
    "INSERT INTO players (player_id, first_name, family_name) VALUES (1, 'a', 'b')",
    "UPDATE players SET first_name = 'x'",
    "DELETE FROM players",
    "DROP TABLE players",
    "DROP TABLE IF EXISTS players",
    "ALTER TABLE players ADD COLUMN pwned TEXT",
    "ALTER TABLE players RENAME TO pwned",
    "CREATE TABLE pwned (x INTEGER)",
    "CREATE TABLE pwned AS SELECT * FROM players",
    "CREATE VIEW pwned AS SELECT * FROM players",
    "CREATE INDEX pwned ON players (first_name)",
    "CREATE TRIGGER pwned AFTER INSERT ON players BEGIN DELETE FROM players; END",
    "REPLACE INTO players (player_id, first_name, family_name) VALUES (1, 'a', 'b')",
    "INSERT OR REPLACE INTO players (player_id, first_name, family_name) VALUES (1, 'a', 'b')",
    "WITH x AS (SELECT 1) INSERT INTO players (player_id, first_name, family_name) SELECT 9, 'a', 'b' FROM x",
    "WITH x AS (SELECT 1) DELETE FROM players",
    "UPDATE players SET first_name = (SELECT 'x') WHERE player_id IN (SELECT player_id FROM players)",
    "DELETE FROM players WHERE player_id IN (SELECT player_id FROM players)",
    "PRAGMA writable_schema = 1",
    "PRAGMA journal_mode = DELETE",
    "PRAGMA user_version = 99",
    "ATTACH DATABASE ':memory:' AS evil",
    "ATTACH DATABASE 'pwned.db' AS evil",
    "REINDEX",
    "ANALYZE",
    "BEGIN; DELETE FROM players; COMMIT",
    "SAVEPOINT a; DELETE FROM players",
    "INSERT INTO players SELECT * FROM players",
    "SELECT 1; DROP TABLE players",
    "SELECT 1; DELETE FROM players;",
    "/* harmless */ DELETE /* comment */ FROM players",
    "-- select\nDELETE FROM players",
    "delete\tfrom\nplayers",
    "DeLeTe FROM players",
]

ABUSE_ATTACKS = [
    "",
    "   ",
    ";",
    "SELECT",
    "SELECT * FROM",
    "SELECT 1; SELECT 2",
    "SELECT load_extension('evil.dll')",
    "SELECT readfile('C:/Windows/win.ini')",
    "SELECT writefile('pwned.txt', 'x')",
    "SELECT * FROM sqlite_master",
    "SELECT * FROM sqlite_schema",
    "SELECT * FROM sqlite_temp_master",
    "SELECT name FROM pragma_table_info('players')",
    "SELECT * FROM pragma_database_list",
    "SELECT * FROM nonexistent_table",
    "SELECT fts3_tokenizer('simple')",
    "SELECT randomblob(1000000000)",
    "SELECT zeroblob(2000000000)",
    "SELECT hex(zeroblob(1000000000))",
    "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT count(*) FROM r",
    "SELECT count(*) FROM lineup_stats a, lineup_stats b, lineup_stats c, lineup_stats d, lineup_stats e",
    "EXPLAIN SELECT * FROM players",
    "VACUUM",
    "VACUUM INTO 'copy.db'",
    "PRAGMA table_info(players)",
    "DETACH DATABASE main",
]
