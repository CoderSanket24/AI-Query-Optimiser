package com.aioptimizer.middleware.security;

import net.sf.jsqlparser.JSQLParserException;
import net.sf.jsqlparser.parser.CCJSqlParserUtil;
import net.sf.jsqlparser.statement.select.Select;
import net.sf.jsqlparser.statement.Statement;
import org.springframework.stereotype.Service;

@Service
public class AstFirewall {
    public boolean inspectQuery(String rawSql){
        try {
            Statement statement = CCJSqlParserUtil.parse(rawSql);

            if (!(statement instanceof Select)){
                throw new SecurityException("Blocked: Only SELECT queries are allowed.");
            }

            String upperSql = rawSql.toUpperCase();
            if (upperSql.contains("1=1") || upperSql.contains("1 = 1") || upperSql.contains("OR TRUE")){
                throw new SecurityException("Blocked: Tautology detected (Possible SQL Injection).");
            }

            System.out.println("AST Firewall: Query is structurally safe.");
            return true;
        }catch (JSQLParserException e){
            throw new SecurityException("Blocked: Malformed SQL syntax.");
        }
    }
}
