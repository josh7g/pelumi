llm_data = {
    'findings': [{
        "ID": 1,
        "file": "/tmp/scanner_63jzmf4a/repo_20241202_204253/BrokenAuth/broken-auth-spoof/vsnippet/5-broken-auth-spoof.go",
        "code_snippet": "\thttp.ListenAndServe(addr, nil)",
        "message": "Found an HTTP server without TLS. Use 'http.ListenAndServeTLS' instead. See https://golang.org/pkg/net/http/#ListenAndServeTLS for more information.",
        "severity": "WARNING"
    }, {
        "ID": 2,
        "file": "/tmp/scanner_63jzmf4a/repo_20241202_204253/BufferOverflow/bof-classic/vsnippet/23-bof-classic.c",
        "code_snippet": "        gets(tryOTP);",
        "message": "Avoid 'gets()'. This function does not consider buffer boundaries and can lead to buffer overflows. Use 'fgets()' or 'gets_s()' instead.",
        "severity": "ERROR"
    }, {
        "ID": 3,
        "file": "/tmp/scanner_63jzmf4a/repo_20241202_204253/BusinessLogic/business-logic-money-transfer/vsnippet/28-business-logic-money-transfer.py",
        "code_snippet": "    app.run(host='0.0.0.0', port=1337, debug=True)",
        "message": "Running flask app with host 0.0.0.0 could expose the server publicly.",
        "severity": "WARNING"
    }, {
        "ID": 4,
        "file": "/tmp/scanner_63jzmf4a/repo_20241202_204253/BusinessLogic/business-logic-money-transfer/vsnippet/28-business-logic-money-transfer.py",
        "code_snippet": "    app.run(host='0.0.0.0', port=1337, debug=True)",
        "message": "Detected Flask app with debug=True. Do not deploy to production with this flag enabled as it will leak sensitive information. Instead, consider using Flask configuration variables or setting 'debug' using system environment variables.",
        "severity": "WARNING"
    }, {
        "ID": 5,
        "file": "/tmp/scanner_63jzmf4a/repo_20241202_204253/BusinessLogic/business-logic-money-transfer/vsnippet/templates/index.html",
        "code_snippet": "<h2> Message: {{ message | safe }} </h2>",
        "message": "Detected a segment of a Flask template where autoescaping is explicitly disabled with '| safe' filter. This allows rendering of raw HTML in this segment. Ensure no user data is rendered here, otherwise this is a cross-site scripting (XSS) vulnerability.",
        "severity": "WARNING"
    }, {
        "ID": 6,
        "file": "/tmp/scanner_63jzmf4a/repo_20241202_204253/CachePoisoning/27-cache-poisoning-classic.py/vsnippet/27-cache-poisoning-classic.py",
        "code_snippet": "    app.run(host='0.0.0.0', port=1337, debug=True)",
        "message": "Running flask app with host 0.0.0.0 could expose the server publicly.",
        "severity": "WARNING"
    }, {
        "ID": 7,
        "file": "/tmp/scanner_63jzmf4a/repo_20241202_204253/CachePoisoning/27-cache-poisoning-classic.py/vsnippet/27-cache-poisoning-classic.py",
        "code_snippet": "    app.run(host='0.0.0.0', port=1337, debug=True)",
        "message": "Detected Flask app with debug=True. Do not deploy to production with this flag enabled as it will leak sensitive information. Instead, consider using Flask configuration variables or setting 'debug' using system environment variables.",
        "severity": "WARNING"
    }, {
        "ID": 8,
        "file": "/tmp/scanner_63jzmf4a/repo_20241202_204253/CachePoisoning/27-cache-poisoning-classic.py/vsnippet/templates/index.html",
        "code_snippet": "<h2> {{ result | safe }} </h2>",
        "message": "Detected a segment of a Flask template where autoescaping is explicitly disabled with '| safe' filter. This allows rendering of raw HTML in this segment. Ensure no user data is rendered here, otherwise this is a cross-site scripting (XSS) vulnerability.",
        "severity": "WARNING"
    }, {
        "ID": 9,
        "file": "/tmp/scanner_63jzmf4a/repo_20241202_204253/CommandInjection/command-injection-classic/vsnippet/42-command-injection-classic.py",
        "code_snippet": "    app.run(host='0.0.0.0', port=1337, debug=True)",
        "message": "Running flask app with host 0.0.0.0 could expose the server publicly.",
        "severity": "WARNING"
    }, {
        "ID": 10,
        "file": "/tmp/scanner_63jzmf4a/repo_20241202_204253/CommandInjection/command-injection-classic/vsnippet/42-command-injection-classic.py",
        "code_snippet": "    app.run(host='0.0.0.0', port=1337, debug=True)",
        "message": "Detected Flask app with debug=True. Do not deploy to production with this flag enabled as it will leak sensitive information. Instead, consider using Flask configuration variables or setting 'debug' using system environment variables.",
        "severity": "WARNING"
    }]
}
