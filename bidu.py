import ast
import builtins
import collections
import collections.abc
import functools
import http.cookies
import importlib.machinery
import inspect
import itertools
import pathlib
import re
import sys
import types
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
    EXCLUDE = dir(builtins)

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
                    args=[ast.arg(arg="ctx")],
                    vararg=None,
                    kwonlyargs=[
                        ast.arg(arg=x) for x in names
                    ],
                    kw_defaults=[ast.Constant(value=None) for x in names],
                    kwarg=None,
                    defaults=[ast.Constant(value=None)],
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

    def _exec(self, path, namespace):
        path = pathlib.Path(path)
        text = path.read_text(encoding='UTF-8')
        tree = parse(tokenize(text))
        exec(compile(tree, path.name, 'exec'), {}, namespace)

    def load_mod(self, path):
        path = pathlib.Path(path)
        if path.name not in self.cache:
            mod = types.ModuleType(path.stem)
            self._exec(path, mod.__dict__)
            self.cache[path.name] = mod
        return self.cache[path.name]
    
    def load(self, path):
        return self.load_mod(path).render

    def render(self, path, **kwargs):
        tmpl = self.load(path)
        return tmpl(**kwargs)

import importlib
import importlib.abc
import importlib.util
import sys

class HTMLFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if path is None or path == "":
            path = [pathlib.Path.cwd()]
        if "." in fullname:
            *parents, name = fullname.split(".")
        else:
            name = fullname
        for entry in path:
            p = pathlib.Path(entry, name)
            
            if p.is_dir():
                filename = p / "__init__.tmpl"
                submodule_locations = [p]
            else:
                filename = p.with_suffix(".tmpl")
                submodule_locations = None
            if not filename.exists():
                continue

            return importlib.util.spec_from_file_location(
                fullname, filename, loader=HTMLLoader(filename),
                submodule_search_locations=submodule_locations,
            )

class HTMLLoader(importlib.abc.Loader):
    def __init__(self, filename):
        self.filename = filename
        self._templates = TemplateLoader()

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self._templates._exec(self.filename, module.__dict__)

sys.meta_path.append(HTMLFinder())


class Request(collections.abc.Mapping):
    def __init__(self, environ, *posargs, **kwargs):
        super().__init__(*posargs, **kwargs)
        self.environ = {
            k: v 
            for k, v in environ.items()
            if k.startswith('HTTP_') or k.startswith('wsgi.') or k in (
                "REQUEST_METHOD",
                "SCRIPT_NAME",
                "PATH_INFO",
                "QUERY_STRING",
                "CONTENT_TYPE",
                "CONTENT_LENGTH",
                "SERVER_NAME",
                "SERVER_PORT",
                "SERVER_PROTOCOL",
            )
        }
        self.request_method = self.environ['REQUEST_METHOD']
        self.path_info = self.environ['PATH_INFO']
        self.query = urllib.parse.parse_qs(self.environ.get('QUERY_STRING', ''))

    def __getitem__(self, key):
        return self.environ[key]

    def __iter__(self):
        return iter(self.environ)

    def __len__(self):
        return len(self.environ)
    
    @property
    def cookies(self):
        if not "HTTP_COOKIE" in self.environ:
            return {}
        return http.cookies.SimpleCookie(self.environ["HTTP_COOKIE"])


class RuleSegment:
    def __init__(self, name, pattern, type, type_name):
        self.name = name
        self.pattern = pattern
        self.type = type
        self.type_name = type_name

    def match(self, segment):
        if (m := self.pattern.fullmatch(segment)):
            return {k: self.type(v) for k, v in m.groupdict().items()}


class Rule:
    def __init__(self, method, route, handler, endpoint_name=None):
        self.method = method
        self.route = route
        self.handler = handler
        self.endpoint_name = endpoint_name if endpoint_name is not None else handler.__name__
        self._handler_parameters = inspect.signature(self.handler).parameters
        self._segments = self._parse_route()
        self.count = len([s for s in self._segments if isinstance(s, RuleSegment)])

    def __call__(self, request):
        func = self.match_path(request.path_info)
        return func(request)

    def __eq__(self, other):
        if isinstance(other, Rule):
            return (
                (self.method, self.route, self.handler, self.endpoint_name) == 
                (other.method, other.route, other.handler, other.endpoint_name)
            )
        if isinstance(other, Request):
            return (self.method == other.request_method) and bool(self.match_path(other.path_info))
        if isinstance(other, str):
            return bool(self.match_path(other))
        return NotImplemented
    
    def redirect(self, **kwargs):
        new_path = []
        for segment in self._segments:
            if isinstance(segment, str):
                new_path.append(segment)
                continue
            new_path.append(kwargs.pop(segment.name))
        return str(pathlib.PurePosixPath(*new_path))

    def match_path(self, other):
        parts = pathlib.PurePosixPath(other).parts
        kwargs = {}
        if len(parts) != len(self._segments):
            return
        for part, segment in zip(parts, self._segments):
            if isinstance(segment, str):
                if part != segment:
                    return
                continue
            if isinstance(segment, RuleSegment):
                if (m := segment.match(part)):
                    kwargs.update(m)
                else:
                    return
        return functools.partial(self.handler, **kwargs)


    def _parse_route(self):
        type_patterns = {
            'str': '[^/]+',
            'int': r'\d+',
        }
        types = {
            'str': str,
            'int': int,
        }
        parts = pathlib.PurePosixPath(self.route).parts[1:]
        segments = ['/']
        for part in parts:
            if not part.startswith('<') and not part.endswith('>'):
                segments.append(part)
                continue
            part = part.removeprefix('<').removesuffix('>')
            type, _, name = part.partition(':')
            if not name:
                name = type
                type = 'str'
            if name not in self._handler_parameters:
                raise ValueError(f"Name {name!r} not defined on {self.handler!r}")
            pattern = re.compile(f"^(?P<{name}>{type_patterns.get(type)})$")
            segments.append(RuleSegment(name, pattern, types[type], type))
        return segments

class Response(collections.abc.MutableMapping):
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
        self._headers = dict(headers) if headers is not None else {}
        self._body = body
        self.encoding = encoding
        self.content_type = content_type
        self.cookies = http.cookies.SimpleCookie()

    def __getitem__(self, name):
        return self._headers[name]

    def __setitem__(self, name, value):
        self._headers[name] = value

    def __delitem__(self, name):
        del self._headers[name]

    def __iter__(self):
        return iter(self._headers)
    
    def __len__(self):
        return len(self._headers)

    @property
    def status(self):
        if self._status is None:
            self._status = 200
        if isinstance(self._status, int):
            return self.STATUS_MAP.get(self._status, '200 OK')
        return self._status

    @property
    def headers(self):
        cookies = [("Set-Cookie", morsel.OutputString()) for morsel in self.cookies.values()]
        if 'Content-Type' not in self._headers:
            self._headers['Content-Type'] = f'{self.content_type}; charset=utf-8'
        headers = [*self._headers.items(), *cookies]
        return headers

    @property
    def body(self):
        if isinstance(self._body, str):
            return [self._body.encode(self.encoding)]
        if isinstance(self._body, bytes):
            return [self._body]
        return iter(self._body)


HTTP_404 = Response("Not Found", status=404)

class Application:
    def __init__(self):
        self.routes = []
        self.endpoints = collections.defaultdict(list)
        self.templates = TemplateLoader()

    def url_for(self, endpoint_name, **kwargs):
        for rule in self.endpoints[endpoint_name]:
            try:
                if len(kwargs) != rule.count:
                    continue
                return rule.redirect(**kwargs)
            except KeyError:
                pass

    def router(self, func=None, /, *, method=None, route=None, endpoint_name=None):
        if func is None:
            return functools.partial(self.router, method=method, route=route, endpoint_name=endpoint_name)
        rule = Rule(method, route, func, endpoint_name)
        self.routes.append(rule)
        self.endpoints[rule.endpoint_name].append(rule)
        return func

    get = functools.partialmethod(router, method='GET')
    post = functools.partialmethod(router, method='POST')
    put = functools.partialmethod(router, method='PUT')
    patch = functools.partialmethod(router, method='PATCH')
    delete = functools.partialmethod(router, method='DELETE')

    def dispatch(self, request):
        for route in self.routes:
            if request != route:
                continue
            return route(request)
        return HTTP_404

    def __call__(self, environ, start_response):
        request = Request(environ)
        response = self.dispatch(request)
        if isinstance(response, (str, bytes)):
            response = Response(response)
        start_response(response.status, response.headers)
        self._active_request = None
        return response.body
