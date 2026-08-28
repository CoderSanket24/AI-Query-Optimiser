package com.aioptimizer.middleware.service;

import net.sf.jsqlparser.parser.CCJSqlParserUtil;
import net.sf.jsqlparser.statement.Statement;
import net.sf.jsqlparser.statement.select.Select;
import net.sf.jsqlparser.util.TablesNamesFinder;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class QueryFeatureExtractor {
    public List<String> extractTables(String rawSql) throws Exception {
        Statement statement = CCJSqlParserUtil.parse(rawSql);
        Select selectStatement = (Select) statement;

        TablesNamesFinder tablesNamesFinder = new TablesNamesFinder();
        return tablesNamesFinder.getTableList(selectStatement);
    }
}
