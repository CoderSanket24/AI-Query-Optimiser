package com.aioptimizer.middleware.service;

import net.sf.jsqlparser.expression.BinaryExpression;
import net.sf.jsqlparser.expression.Expression;
import net.sf.jsqlparser.parser.CCJSqlParserUtil;
import net.sf.jsqlparser.schema.Column;
import net.sf.jsqlparser.schema.Table;
import net.sf.jsqlparser.statement.Statement;
import net.sf.jsqlparser.statement.select.*;
import org.springframework.stereotype.Service;

import java.util.*;

/**
 * QueryFeatureExtractor
 * ---------------------
 * Walks the JSQLParser AST to extract per-table alias/name pairs AND
 * join-edge pairs from every FROM item and JOIN item in the query.
 *
 * TableRef  – (alias, realName) for one table reference.
 *   alias     = identifier pg_hint_plan sees (falls back to realName if no AS)
 *   realName  = physical table name used for schema stat lookups
 *
 * JoinEdge  – (leftAlias, rightAlias) for one explicit equi-join predicate.
 *   Extracted from ON conditions of JOIN clauses.
 *   Used by the Python policy to build a join graph and avoid disconnected
 *   Leading hints that would force cross-product prefixes.
 *
 * Self-joins are supported: "orders o1 JOIN orders o2 ON o1.id = o2.ref_id"
 * produces two TableRefs with realName="orders" but distinct aliases.
 */
@Service
public class QueryFeatureExtractor {

    /** Immutable pair: (alias-or-name for hints, real table name for stats). */
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

    /** An explicit join predicate between two table aliases (from ON clause). */
    public static class JoinEdge {
        public final String leftAlias;
        public final String rightAlias;

        public JoinEdge(String leftAlias, String rightAlias) {
            this.leftAlias  = leftAlias;
            this.rightAlias = rightAlias;
        }

        @Override public String toString() {
            return leftAlias + "↔" + rightAlias;
        }
    }

    /** Combined extraction result. */
    public static class ExtractionResult {
        public final List<TableRef> tableRefs;
        public final List<JoinEdge> joinEdges;

        public ExtractionResult(List<TableRef> tableRefs, List<JoinEdge> joinEdges) {
            this.tableRefs = Collections.unmodifiableList(tableRefs);
            this.joinEdges = Collections.unmodifiableList(joinEdges);
        }
    }

    // ── Public API ──────────────────────────────────────────────────────────────

    /**
     * Full extraction: table refs + join edges.
     */
    public ExtractionResult extract(String rawSql) throws Exception {
        Statement statement = CCJSqlParserUtil.parse(rawSql);
        if (!(statement instanceof Select)) {
            throw new IllegalArgumentException("Only SELECT statements are supported");
        }
        SelectBody body = ((Select) statement).getSelectBody();

        List<TableRef> refs  = new ArrayList<>();
        List<JoinEdge> edges = new ArrayList<>();
        collectFromBody(body, refs, edges);
        return new ExtractionResult(refs, edges);
    }

    /**
     * Alias-list shim: returns just the alias strings (hint identifiers).
     * Used by code that only needs aliases.
     */
    public List<String> extractTables(String rawSql) throws Exception {
        ExtractionResult r = extract(rawSql);
        List<String> aliases = new ArrayList<>();
        for (TableRef t : r.tableRefs) aliases.add(t.alias);
        return aliases;
    }

    /**
     * Legacy shim: extracts TableRef list only.
     */
    public List<TableRef> extractTableRefs(String rawSql) throws Exception {
        return extract(rawSql).tableRefs;
    }

    // ── AST walking ──────────────────────────────────────────────────────────────

    private void collectFromBody(SelectBody body, List<TableRef> refs, List<JoinEdge> edges) {
        if (body instanceof PlainSelect) {
            collectFromPlainSelect((PlainSelect) body, refs, edges);
        } else if (body instanceof SetOperationList) {
            for (SelectBody sub : ((SetOperationList) body).getSelects()) {
                collectFromBody(sub, refs, edges);
            }
        }
    }

    private void collectFromPlainSelect(PlainSelect ps, List<TableRef> refs, List<JoinEdge> edges) {
        // Build alias→tableRef map so we can resolve ON-clause column references
        Map<String, String> aliasMap = new LinkedHashMap<>();

        // FROM item
        FromItem from = ps.getFromItem();
        if (from != null) {
            collectFromItem(from, refs, aliasMap);
        }

        // JOIN items
        List<Join> joins = ps.getJoins();
        if (joins != null) {
            for (Join join : joins) {
                FromItem right = join.getRightItem();
                collectFromItem(right, refs, aliasMap);

                // Extract join edge from ON expression
                if (join.getOnExpression() != null) {
                    collectEdgesFromExpression(join.getOnExpression(), aliasMap, edges);
                }
            }
        }
    }

    private void collectFromItem(FromItem item, List<TableRef> refs, Map<String, String> aliasMap) {
        if (item instanceof Table) {
            Table  table    = (Table) item;
            String realName = table.getName();
            String alias    = (table.getAlias() != null)
                              ? table.getAlias().getName()
                              : realName;
            refs.add(new TableRef(alias, realName));
            aliasMap.put(alias.toLowerCase(), alias);   // for edge resolution

        } else if (item instanceof SubSelect) {
            SubSelect sub = (SubSelect) item;
            if (sub.getSelectBody() != null) {
                // Subqueries are opaque — walk but don't map aliases to parent scope
                List<TableRef> subRefs  = new ArrayList<>();
                List<JoinEdge> subEdges = new ArrayList<>();
                collectFromBody(sub.getSelectBody(), subRefs, subEdges);
                refs.addAll(subRefs);
            }

        } else if (item instanceof ParenthesedFromItem) {
            collectFromItem(((ParenthesedFromItem) item).getFromItem(), refs, aliasMap);
        }
    }

    /**
     * Walk an ON expression tree and emit a JoinEdge for every equi-join
     * predicate of the form  tableA.col = tableB.col.
     */
    private void collectEdgesFromExpression(Expression expr,
                                            Map<String, String> aliasMap,
                                            List<JoinEdge> edges) {
        if (expr instanceof BinaryExpression) {
            BinaryExpression be = (BinaryExpression) expr;
            String op = be.getStringExpression();

            if ("=".equals(op)) {
                // Could be a column equi-join
                String leftTable  = tableOfExpr(be.getLeftExpression());
                String rightTable = tableOfExpr(be.getRightExpression());
                if (leftTable != null && rightTable != null
                        && !leftTable.equalsIgnoreCase(rightTable)) {
                    // Resolve to canonical alias casing
                    String la = aliasMap.getOrDefault(leftTable.toLowerCase(),  leftTable);
                    String ra = aliasMap.getOrDefault(rightTable.toLowerCase(), rightTable);
                    edges.add(new JoinEdge(la, ra));
                }
            } else if ("AND".equalsIgnoreCase(op)) {
                // Recurse into compound predicates
                collectEdgesFromExpression(be.getLeftExpression(),  aliasMap, edges);
                collectEdgesFromExpression(be.getRightExpression(), aliasMap, edges);
            }
        }
    }

    /** Return the table/alias qualifier of a Column expression, or null. */
    private String tableOfExpr(Expression expr) {
        if (expr instanceof Column) {
            Table t = ((Column) expr).getTable();
            return (t != null) ? t.getName() : null;
        }
        return null;
    }
}
