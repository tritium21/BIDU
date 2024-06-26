import ast
import collections
import collections.abc
import functools
import pathlib
import re
import urllib.parse


class Visit:
    def __init__(self, funcname):
        self.names = []
        self.funcname = funcname

    def _yield(self, node):
        return ast.Expr(ast.Yield(node))

    def _string(self, node):
        return ast.Call(
            func=ast.Name('str', ctx=ast.Load()),
            args=[node],
            keywords=[],
        )

    def visit(self, node):
        if not isinstance(node, Node):
            raise TypeError()
        node_name = node.__class__.__name__
        method = getattr(self, f'visit_{node_name}', self.visit_generic)
        return method(node)

    def visit_generic(self, node):
        raise ValueError(f"Unknown node type {node.__class__.__name__!r}")

    def visit_TemplateNode(self, node):
        _inner_body = [self.visit(x) for x in node.value]
        body = [
            ast.FunctionDef(
                name='_inner',
                args=ast.arguments(
                    posonlyargs=[],
                    args=[],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[]
                ),
                body=_inner_body,
                decorator_list=[],
                returns=None
            ),
            ast.Return(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Constant(value=''),
                        attr='join',
                        ctx=ast.Load()
                    ),
                    args=[
                        ast.Call(
                            func=ast.Name(id='_inner', ctx=ast.Load()),
                            args=[],
                            keywords=[]
                        )
                    ],
                    keywords=[]
                )
            )
        ]
        tree = ast.Interactive(body=[
            ast.FunctionDef(
                name=self.funcname,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg=n, annotation=None) for n in set(self.names)],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[]
                ),
                body=body,
                decorator_list=[],
                returns=None
            )]
        )
        return ast.fix_missing_locations(tree)

    def visit_StringNode(self, node):
        return self._yield(ast.Constant(value=node.value))

    def visit_NameNode(self, node):
        self.names.append(node.value)
        return self._yield(self._string(ast.Name(id=node.value, ctx=ast.Load())))

    def visit_ForNode(self, node):
        self.names.append(node.name)
        iter_ = ast.Name(id=node.name, ctx=ast.Load())
        target = ast.Name(id=node.target, ctx=ast.Store())
        body = list(self.visit(x) for x in node.value)
        if node.target in self.names:
            del self.names[self.names.index(node.target)]
        orelse = []
        return ast.For(target=target, iter=iter_, body=body, orelse=orelse)


class Node:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.value)})"


class TemplateNode(Node):
    pass


class StringNode(Node):
    pass


class NameNode(Node):
    pass


class ForNode(Node):
    def __init__(self, value, *, name, target):
        self.value = value
        self.name = name
        self.target = target

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value!r}, name={self.name!r}, target={self.target!r})"


class Template:
    def __init__(self, source, id=None):
        self._cache = None
        self.source = source

    @classmethod
    def from_file(cls, path):
        path = pathlib.Path(path)
        return cls(path.read_text(encoding='utf-8'))

    @staticmethod
    def tokenize(instring):
        toks = iter(re.split(r'(({{|{%)(.*?)(%}|}}))', instring))
        while True:
            try:
                tok = next(toks)
            except StopIteration:
                break
            if not tok.startswith('{{') and not tok.startswith('{%'):
                yield ('text', tok)
                continue
            open_brace = next(toks)
            exp = next(toks).strip()
            next(toks)
            if open_brace == '{{':
                yield ('name', exp)
                continue
            yield ('statement', exp)

    @staticmethod
    def parse(stream):
        def _parse(stream, end=None):
            for token in stream:
                if token[0] == 'statement' and token[1] == end:
                    break
                if token[0] == 'text':
                    yield StringNode(token[1])
                    continue
                if token[0] == 'name':
                    yield NameNode(token[1])
                    continue
                kind = token[1].partition(' ')[0]
                endkind = f'end{kind}'
                if kind == 'for':
                    match = re.match(r"^for\s+(.*?)\s+in\s+(.*?)$", token[1])
                    if not match:
                        raise ValueError(f"Parse Error: {token[1]!r}")
                    target, name = match.groups()
                    name = name
                    yield ForNode(
                        list(_parse(stream, end=endkind)),
                        name=name,
                        target=target
                    )
        return TemplateNode(list(_parse(stream)))

    def _build(self):
        namespace = {}
        visitor = Visit('template')
        tokens = self.tokenize(self.source)
        parse_tree = self.parse(tokens)
        ast_tree = visitor.visit(parse_tree)
        exec(compile(ast_tree, '<template>', 'single'), {}, namespace)
        return namespace.get('template')

    def render(self, *ctx, **kwctx):
        if self._cache is None:
            self._cache = self._build()
        return self._cache(*ctx, **kwctx)


class Request(collections.abc.Mapping):
    KEYS = '''REQUEST_METHOD SCRIPT_NAME PATH_INFO
    QUERY_STRING CONTENT_TYPE CONTENT_LENGTH
    SERVER_NAME SERVER_PORT SERVER_PROTOCOL wsgi.version
    wsgi.url_scheme wsgi.input wsgi.errors wsgi.multithread
    wsgi.multiprocess wsgi.run_once'''.strip().split()

    def __init__(self, environ):
        self.environ = {
            k: v for k, v in environ.items()
            if k.startswith('HTTP_') or k in self.KEYS
        }
        self.path_info = self.environ['PATH_INFO']
        self.script_name = self.environ['SCRIPT_NAME']
        self.method = self.environ['REQUEST_METHOD']
        self.scheme = self.environ['wsgi.url_scheme']
        self.query = urllib.parse.parse_qs(self.environ['QUERY_STRING'])

    def __getitem__(self, key):
        return self.environ[key]

    def __iter__(self):
        return iter(self.environ)

    def __len__(self):
        return len(self.environ)

    @property
    def netloc(self):
        scheme = self['wsgi.url_scheme']
        if self.get('HTTP_HOST'):
            netloc = self['HTTP_HOST']
        else:
            netloc = self['SERVER_NAME']
            port = self['SERVER_PORT']
            if (
                (scheme == 'https' and port != '443') or
                (scheme == 'http' and port != '80')
            ):
                netloc += ':' + port
        return netloc

    @property
    def root(self):
        parts = [
            self.scheme,
            self.netloc,
            urllib.parse.quote(self.script_name),
            None, None
        ]
        return urllib.parse.urlunsplit(parts)

    @property
    def this(self):
        parts = [self.scheme]
        path = urllib.parse.quote(self.script_name + self.path_info)
        parts.extend([self.netloc, path])
        if self.query:
            parts.extend(
                [urllib.parse.urlencode(self.query, doseq=True), None]
            )
        else:
            parts.extend([None, None])
        return urllib.parse.urlunsplit(parts)


class Response:
    STATUS_MAP = {
        200: "200 OK",
        201: "201 Created",
        202: "202 Accepted",
        204: "204 No Content",
        301: "301 Moved Permanently",
        302: "302 Moved Temporarily",
        304: "304 Not Modified",
        400: "400 Bad Request",
        401: "401 Unauthorized",
        403: "403 Forbidden",
        404: "404 Not Found",
        500: "500 Internal Server Error",
        501: "501 Not Implemented",
        502: "502 Bad Gateway",
        503: "503 Service Unavailable",
    }

    def __init__(self, body, *, status=None, headers=None, encoding='utf-8', content_type='text/html'):
        self._status = status
        self._headers = headers if headers is not None else {}
        self._body = body
        self.encoding = encoding
        self.content_type = content_type

    @property
    def status(self):
        if self._status is None:
            self._status = 200
        if isinstance(self._status, int):
            return self.STATUS_MAP.get(self._status, '200 OK')
        return self._status

    @property
    def headers(self):
        if isinstance(self._headers, (dict, collections.abc.Mapping)):
            if 'Content-Type' not in self._headers:
                self._headers['Content-Type'] = f'{self.content_type}; charset=utf-8'
            return list(self._headers.items())
        return self._headers

    @property
    def body(self):
        if isinstance(self._body, str):
            return [self._body.encode(self.encoding)]
        if isinstance(self._body, bytes):
            return [self._body]
        return iter(self._body)


HTTP_404 = Response("Not Found", status=404)


class Rule:
    def __init__(self, route, func, endpoint=None):
        self.route = route
        self.count = None
        self.pattern = None
        self.types = None
        self.redirect_pattern = None
        self.endpoint = endpoint if endpoint is not None else func.__name__
        self.function = func
        self._parse_route()

    def __eq__(self, other):
        if not isinstance(other, (str, Request, Rule)):
            return False
        if isinstance(other, Rule):
            return (
                (self.route, self.types, self.endpoint, self.function) ==
                (other.route, other.types, other.endpoint, other.function)
            )
        elif isinstance(other, str):
            match, redirect = self.match(other)
        else:
            match, redirect = self.match(other.path_info)
        if match or redirect:
            return True
        return False

    def match(self, other):
        return self.pattern.match(other), self.redirect_pattern.match(other)

    def __call__(self, request):
        match, redirect = self.match(request.path_info)
        matches = match.groupdict()
        kwargs = {k: self.types[k](v) for k, v in matches.items()}
        return self.function(request, **kwargs)

    def _parse_route(self):
        route = self.route
        type_map_re = {'int': r'\d+', 'str': '[^/]+'}
        type_map = {'int': int, 'str': str}
        in_arg = False
        in_arg_name = False
        arg_name = ''
        type_name = ''
        pat = '^'
        replace = ''
        types = {}
        count = 0
        for c in route:
            if not in_arg and c != '<':
                pat += c
                replace += c
                continue
            if not in_arg and c == '<':
                count += 1
                in_arg = True
                in_arg_name = True
                pat += '(?P<'
                replace += '{'
                continue
            if in_arg and in_arg_name and c not in ':>':
                arg_name += c
                continue
            if in_arg and in_arg_name and c == ':':
                pat += arg_name + '>'
                replace += arg_name + '}'
                in_arg_name = False
                continue
            if in_arg and in_arg_name and c == '>':
                in_arg = False
                in_arg_name = False
                pat += arg_name + '>' + type_map_re['str'] + ')'
                types[arg_name] = type_map['str']
                replace += arg_name + '}'
                arg_name = ''
                continue
            if in_arg and not in_arg_name and c != '>':
                type_name += c
                continue
            if in_arg and not in_arg_name and c == '>':
                in_arg = False
                pat += type_map_re[type_name] + ')'
                types[arg_name] = type_map[type_name]
                type_name = ''
                arg_name = ''
                continue
        pat += '$'
        if pat[-2:] == '/$':
            redir = pat[:-2] + '$'
        else:
            redir = pat[:-1] + '/$'
        self.redirect_pattern = re.compile(redir)
        self.pattern = re.compile(pat)
        self.types = types
        self.replace = replace
        self.count = count


class Application:
    def __init__(self):
        self.routes = []
        self.endpoints = collections.defaultdict(list)
        self._active_request = None

    def url_for(self, endpoint, **kwargs):
        root = self._active_request.root
        for endpoint in self.endpoints[endpoint]:
            try:
                if len(kwargs) != endpoint.count:
                    continue
                path_info = endpoint.replace.format(**kwargs)
                return root + path_info
            except KeyError:
                pass

    def router(self, func=None, /, *, route=None, endpoint=None):
        if func is None:
            return functools.partial(self.router, route=route, endpoint=endpoint)
        rule = Rule(route, func, endpoint)
        self.routes.append(rule)
        self.endpoints[rule.endpoint].append(rule)
        return func

    def dispatch(self, request):
        for route in self.routes:
            if request != route:
                continue
            main, redirect = route.match(request.path_info)
            if main:
                return route(request)
            kwargs = redirect.groupdict()
            url = self.url_for(route.endpoint, **kwargs)
            headers = {'Location': url}
            status = 301
            body = b''
            return Response(body, status=status, headers=headers)
        return HTTP_404

    def __call__(self, environ, start_response):
        self._active_request = request = Request(environ)
        response = self.dispatch(request)
        if isinstance(response, (str, bytes)):
            response = Response(response)
        start_response(response.status, response.headers)
        self._active_request = None
        return response.body


if __name__ == '__main__':
    import os
    import getpass
    from wsgiref.simple_server import make_server
    HOST = os.environ.get('SERVER_HOST', '')
    PORT = int(os.environ.get('SERVER_PORT', 8000))

    application = Application()

    @application.router(route='/')
    def root(request):
        return Response('', status=302, headers={'location': application.url_for('hello', bar=getpass.getuser())})

    @application.router(route='/hello/<bar>/')
    @application.router(route='/hello/')
    def hello(request, bar='World'):
        template = Template.from_file('template.html')
        title = f'Hello, {bar}'
        body = "I am the very model of a modern major general!"
        return Response(template.render(title=title, body=body))

    with make_server(HOST, PORT, application) as httpd:
        httpd.serve_forever()
