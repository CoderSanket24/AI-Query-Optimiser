package com.aioptimizer.middleware.controller;

import com.aioptimizer.middleware.security.AstFirewall;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("api/query")
public class QueryController {

    @Autowired
    private AstFirewall astFirewall;

    @PostMapping("/execute")
    public String executeQuery(@RequestBody String sql){
        try {
            astFirewall.inspectQuery(sql);
            return "SUCCESS: Query passed the firewall and is ready for AI optimization.";
        } catch (SecurityException e) {
            return "SECURITY ALERT: " + e.getMessage();
        }
    }
}
