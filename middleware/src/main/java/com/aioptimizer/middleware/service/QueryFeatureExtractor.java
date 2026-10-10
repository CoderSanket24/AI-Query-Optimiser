package com.aioptimizer.middleware.service;

import net.sf.jsqlparser.parser.CCJSqlParserUtil;
import net.sf.jsqlparser.schema.Table;
import net.sf.jsqlparser.statement.Statement;
import net.sf.jsqlparser.statement.select.*;
import org.springframework.stereotype.Service;

import java.util.*;

/**
 * QueryFeatureExtractor
 * ---------------------
 * Walks the JSQLParser AST to extract per-table alias/name pairs from every
 * FROM item and JOIN item in the query.
 *
 * Returns a List of TableRef objects, each holding:
 *   alias     – the identifier pg_hint_plan sees in the hinted SQL
 *               (falls back to real table name when no alias is declared)
 *   realName  – the physical table name used to look up schema stats
 *
 * Self-joins are supported: "orders o1 JOIN orders o2" produces two entries
 * with realName="orders" but distinct aliases "o1" and "o2".
 *
 * The flat table-name list forwarded to Python becomes the alias list, so
 * pg_hint_plan Leading hints match what the query actually uses.
 */
@Service
public class QueryFeatureExtractor {

    /** Immutable pair: (alias-or-name used in hint, real table name for stats) */
    public static class TableRef {
        public final String alias;    // identifier pg_hint_plan matches
        public final String realName; // physical name for schema statistics

        public TableRef(String alias, String realName) {
            this.alias    = alias;
            this.realName = realName;
        }

        @Override public String toString() {
            return alias.equals(realName) ? realName : realName + " AS " + alias;
        }
    }

    /**
     * Extract alias→realName pairs from every FROM and JOIN item.
     * Deduplication is intentionally NOT performed: self-joins produce two
     * entries with the same realName but different aliases.
     */
    public List<TableRef> extractTableRefs(String rawSql) throws Exception {
        Statement statement = CCJSqlParserUtil.parse(rawSql);
        if (!(statement instanceof Select)) {
            throw new IllegalArgumentException("Only SELECT statements are supported");
        }
        SelectBody body = ((Select) statement).getSelectBody();

        List<TableRef> refs = new ArrayList<>();
        collectFromBody(body, refs);
        return refs;
    }

    /**
     * Legacy shim: returns just the alias strings (pg_hint_plan identifiers).
     * Used by code that only needs the hint-side identifiers.
     */
    public List<String> extractTables(String rawSql) throws Exception {
        List<TableRef> refs = extractTableRefs(rawSql);
        List<String> aliases = new ArrayList<>();
        for (TableRef r : refs) aliases.add(r.alias);
        return aliases;
    }

    // ── AST walking ──────────────────────────────────────────────────────────

    private void collectFromBody(SelectBody body, List<TableRef> refs) {
        if (body instanceof PlainSelect) {
            collectFromPlainSelect((PlainSelect) body, refs);
        } else if (body instanceof SetOperationList) {
            for (SelectBody sub : ((SetOperationList) body).getSelects()) {
                collectFromBody(sub, refs);
            }
        }
        // SubSelect / WithItem: walk recursively if needed
    }

    private void collectFromPlainSelect(PlainSelect ps, List<TableRef> refs) {
        // FROM item
        FromItem from = ps.getFromItem();
        if (from != null) collectFromItem(from, refs);

        // JOIN items
        List<Join> joins = ps.getJoins();
        if (joins != null) {
            for (Join join : joins) {
                collectFromItem(join.getRightItem(), refs);
            }
        }
    }

    private void collectFromItem(FromItem item, List<TableRef> refs) {
        if (item instanceof Table) {
            Table table = (Table) item;
            String realName = table.getName();
            String alias    = (table.getAlias() != null)
                              ? table.getAlias().getName()
                              : realName;
            refs.add(new TableRef(alias, realName));

        } else if (item instanceof SubSelect) {
            // Subquery — walk its body; the alias belongs to the subquery,
            // not an individual table, so just recurse into it.
            SubSelect sub = (SubSelect) item;
            if (sub.getSelectBody() != null) {
                collectFromBody(sub.getSelectBody(), refs);
            }

        } else if (item instanceof ParenthesedFromItem) {
            // Parenthesised from item (JSQLParser 4.x+)
            collectFromItem(((ParenthesedFromItem) item).getFromItem(), refs);
        }
        // LateralSubSelect, ValuesList, etc. — not JOIN-ordered, skip
    }
}
