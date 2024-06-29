import ast
import collections
import collections.abc
import functools
import itertools
import pathlib
import re
import urllib.parse


def tokenize(instring):
    toks = iter(re.split(r'({{|{%|%}|}})', instring))
    while True:
        try:
            tok = next(toks)
        except StopIteration:
            break
        match tok:
            case "{{":
                yield ('expression', next(toks).strip())
                next(toks)
            case "{%":
                block_head = next(toks).strip()
                next(toks)
                sp = re.split(r'[\s]+', block_head, maxsplit=1)
                block_type, *rest = sp[0:]
                rest = ' '.join(rest)
                match block_type:
                    case 'for':
                        targets, _, expression = rest.partition(' in ')
                        yield (
                            'for',
                            tuple(t for t in re.split(r'(?:,\s*?)+', targets)),
                            expression.strip()
                        )
                    case 'else':
                        yield ('else',)
                    case 'endfor':
                        yield ('for_end',)
                    case 'set':
                        targets, _, expression = rest.rpartition('=')
                        targets = [
                            tuple(t for t in re.split(r'(?:,\s*?)+', target.strip()))
                            for target in targets.split('=')
                        ]
                        yield ('set', targets, expression.strip())
                    case 'if':
                        yield ('if', rest)
                    case 'endif':
                        yield ('if_end',)
            case _:
                yield ('text', tok)


class NameVisitor(ast.NodeVisitor):
    def __init__(self, *p, **kw):
        super().__init__(*p, **kw)
        self._NAMES = []

    def visit_Name(self, node):
        name = node.id
        if name not in self._NAMES:
            self._NAMES.append(name)

    @classmethod
    def get_names(cls, tree, exclude=()):
        inst = cls()
        inst.visit(tree)
        return [n for n in inst._NAMES if n not in exclude]

def parse(stream):
    ELSE = object()
    EXCLUDE = dir(__builtins__)

    def _expression(expression):
        body = ast.parse(expression).body
        return [
            ast.Expr(ast.Yield(value=ast.Call(
                func=ast.Name('str', ctx=ast.Load()),
                args=[x.value],
                keywords=[],
            )))
            for x in body
        ]
    
    def _target(target):
        if len(target) == 1:
            EXCLUDE.append(target[0])
            return ast.Name(id=target[0], ctx=ast.Store())
        EXCLUDE.extend(target)
        return ast.Tuple(
            elts=[ast.Name(id=t, ctx=ast.Store()) for t in target],
            ctx=ast.Store()
        )

    def _split_else(parse_stream):
        _values = list(parse_stream)
        values = list(itertools.takewhile(lambda x: x is not ELSE, _values))
        orelse = list(itertools.dropwhile(lambda x: x is not ELSE, _values))[1:]
        return values, orelse

    def _parse(stream, end=None):
        stream = iter(stream)
        while True:
            try:
                token = next(stream)
            except StopIteration:
                break
            if token[0] == end:
                return
            match token:
                case ('text', text):
                    yield ast.Expr(ast.Yield(ast.Constant(text)))
                case ('expression', expression):
                    yield from _expression(expression)
                case ('else',):
                    yield ELSE
                case ('for', targets, expression):
                    iter_ = [n.value for n in ast.parse(expression).body][0]
                    target = _target(targets)
                    body, orelse = _split_else(_parse(stream, 'for_end'))
                    yield ast.If(
                        test=iter_,
                        body=[ast.For(target=target, iter=iter_, body=body, orelse=[])],
                        orelse=orelse
                    )
                case ('set', targets, expression):
                    targets = [_target(t) for t in targets]
                    value = [n.value for n in ast.parse(expression).body][0]
                    yield ast.Assign(targets=targets, value=value)
                case ('if', expression):
                    test = [n.value for n in ast.parse(expression).body][0]
                    body, orelse = _split_else(_parse(stream, 'if_end'))
                    yield ast.If(
                        test=test,
                        body=body,
                        orelse=orelse
                    )


    inner_func = ast.FunctionDef(
        name="_inner",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=list(_parse(stream)),
        decorator_list=[],
        returns=None,
    )
    names = NameVisitor.get_names(inner_func, EXCLUDE)

    return ast.fix_missing_locations(ast.Module(
        body=[
            ast.FunctionDef(
                name="render",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[],
                    vararg=None,
                    kwonlyargs=[
                        ast.arg(arg=x) for x in names
                    ],
                    kw_defaults=[None for x in names],
                    kwarg=None,
                    defaults=[],
                ),
                body=[
                    inner_func,
                    ast.Return(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Constant(value=''),
                                attr='join',
                                ctx=ast.Load()
                            ),
                            args=[
                                ast.Call(
                                    func=ast.Name(id="_inner", ctx=ast.Load()),
                                    args=[],
                                    keywords=[],
                                )
                            ],
                            keywords=[]
                        )
                    )
                ],
                decorator_list=[],
                returns=None,
            )

        ],
        type_ignores=[]
    ))

class TemplateLoader:
    def __init__(self):
        self.cache = {}

    def load(self, path):
        path = pathlib.Path(path)
        if path.name not in self.cache:
            text = path.read_text(encoding='UTF-8')
            tree = parse(tokenize(text))
            namespace = {}
            exec(compile(tree, path.name, 'exec'), {}, namespace)
            self.cache[path.name] = namespace['render']
        return self.cache[path.name]

    def render(self, path, **kwargs):
        tmpl = self.load(path)
        return tmpl(**kwargs)


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
        self.templates = TemplateLoader()
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
        title = f'Hello, {bar}'
        body = "I am the very model of a modern major general!"
        items = "This is a list of strings".split()
        return Response(application.templates.render("template.html.tmpl", title=title, body=body, items=items))

    with make_server(HOST, PORT, application) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
