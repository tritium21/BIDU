import ast
import atexit
import builtins
import collections
import collections.abc
import dataclasses
import functools
import http.cookies
import inspect
import itertools
import pathlib
import re
import shelve
import types
import urllib.parse


def tokenize(instring):
    toks = iter(re.split(r'({{|{%|%}|}})', instring))
    while (tok := next(toks, None)) is not None:
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
    # Keeping old EXCLUDE code in place, commented out, in case edge case
    ELSE = object()
    # EXCLUDE = set([])
    DEFAULT_EXCLUDE = set([*dir(builtins), "ctx"])
    INCLUDE = set([])

    def _str(body):
        return [
            ast.Expr(ast.Yield(value=ast.Call(
                func=ast.Name('str', ctx=ast.Load()),
                args=[x.value],
                keywords=[],
            )))
            for x in body
        ]

    def _expression(expression):
        body = ast.parse(expression).body
        includes = itertools.chain.from_iterable(
            NameVisitor.get_names(x, exclude=DEFAULT_EXCLUDE) for x in body
        )
        INCLUDE.update(includes)
        return body
    
    def _target(target):
        if len(target) == 1:
            # EXCLUDE.add(target[0])
            return ast.Name(id=target[0], ctx=ast.Store())
        # EXCLUDE.update(target)
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
        while (token := next(stream, None)) is not None:
            if token[0] == end:
                return
            match token:
                case ('text', text):
                    yield ast.Expr(ast.Yield(ast.Constant(text)))
                case ('expression', expression):
                    yield from _str(_expression(expression))
                case ('else',):
                    yield ELSE
                case ('for', targets, expression):
                    iter_ = [n.value for n in _expression(expression)][0]
                    target = _target(targets)
                    body, orelse = _split_else(_parse(stream, 'for_end'))
                    yield ast.If(
                        test=iter_,
                        body=[ast.For(target=target, iter=iter_, body=body, orelse=[])],
                        orelse=orelse
                    )
                case ('set', targets, expression):
                    targets = [_target(t) for t in targets]
                    value = [n.value for n in _expression(expression)][0]
                    yield ast.Assign(targets=targets, value=value)
                case ('if', expression):
                    test = [n.value for n in _expression(expression)][0]
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
    # names = NameVisitor.get_names(inner_func, (DEFAULT_EXCLUDE|EXCLUDE)-INCLUDE)
    inner_func.body[:] = [ast.Nonlocal(names=list(INCLUDE)), *inner_func.body]
    return ast.fix_missing_locations(ast.Module(
        body=[
            ast.FunctionDef(
                name="render",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg="ctx")],
                    vararg=None,
                    kwonlyargs=[
                        # ast.arg(arg=x) for x in names
                        ast.arg(arg=x) for x in INCLUDE
                    ],
                    # kw_defaults=[ast.Constant(value=None) for x in names],
                    kw_defaults=[ast.Constant(value=None) for x in INCLUDE],
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
    def __init__(self, root=None):
        self.cache = {}
        self.root = pathlib.Path(root) if root is not None else pathlib.Path.cwd()

    @classmethod
    def exec_template(cls, path, namespace):
        path = pathlib.Path(path)
        text = path.read_text(encoding='UTF-8')
        tree = parse(tokenize(text))
        exec(compile(tree, path.name, 'exec'), {}, namespace)

    def load_mod(self, path):
        path = pathlib.Path(path)
        if not path.is_absolute():
            path = self.root / path
        if path.name not in self.cache:
            mod = types.ModuleType(path.stem)
            self.exec_template(path, mod.__dict__)
            self.cache[path.name] = mod
        return self.cache[path.name]
    
    def load(self, path):
        return self.load_mod(path).render

    def bind(self, context):
        def bound(path, *pargs, **kwargs):
            return self.render(path, context, *pargs, **kwargs)
        return bound

    def render(self, path, *pargs, **kwargs):
        tmpl = self.load(path)
        return tmpl(*pargs, **kwargs)


class Request(collections.abc.Mapping):
    def __init__(self, environ, application, *posargs, **kwargs):
        super().__init__(*posargs, **kwargs)
        self.application = application
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

    def url_for(self, endpoint_name, **kwargs):
        for rule in self.application.endpoints[endpoint_name]:
            try:
                if len(kwargs) != rule.count:
                    continue
                return str(pathlib.PurePosixPath(self['SCRIPT_NAME']) / rule.redirect(**kwargs))
            except KeyError:
                pass

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
        return pathlib.PurePosixPath(*pathlib.PurePosixPath(*new_path).parts[1:])

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
        headers = [*self.items(), *cookies]
        return headers

    @property
    def body(self):
        if isinstance(self._body, str):
            return [self._body.encode(self.encoding)]
        if isinstance(self._body, bytes):
            return [self._body]
        return iter(self._body)


HTTP_404 = Response("Not Found", status=404)


class Application(collections.abc.MutableMapping):
    def __init__(self, template_dir=None):
        self.routes = []
        self.context = {}
        self.endpoints = collections.defaultdict(list)
        self.template = TemplateLoader(root=template_dir).bind(self)

    def __getitem__(self, name):
        return self.context[name]

    def __setitem__(self, name, value):
        self.context[name] = value

    def __delitem__(self, name):
        del self.context[name]

    def __iter__(self):
        return iter(self.context)
    
    def __len__(self):
        return len(self.context)

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
        request = Request(environ, self)
        response = self.dispatch(request)
        if isinstance(response, (str, bytes)):
            response = Response(response)
        start_response(response.status, response.headers)
        self._active_request = None
        return response.body

class BaseBackend(collections.abc.MutableMapping):
    def __init__(self, data=None):
        self._data = data
        atexit.register(self.sync)

    def _key(self, key):
        return key

    def __repr__(self):
        data = self._data
        return f"{type(self).__name__}({data=})"

    def __getitem__(self, name):
        return self._data[self._key(name)]

    def __setitem__(self, name, value):
        self._data[self._key(name)] = value

    def __delitem__(self, name):
        del self._data[self._key(name)]
    
    def __len__(self):
        return len(self._data)
    
    def __iter__(self):
        return iter(self._data)

    def close(self):
        raise NotImplementedError()
    
    def sync(self):
        raise NotImplementedError()


class MemoryBackend(BaseBackend):
    def __init__(self, *p, **kw):
        super().__init__(dict(*p, **kw))

    def sync(self):
        return
    
    def close(self):
        return


class ShelveBackend(BaseBackend):
    def __init__(self, filename, flag='c', protocol=None, writeback=False):
        self._filename = filename
        self._flag = flag
        self._protocol = protocol
        self._writeback = writeback
        super().__init__(shelve.open(filename, flag, protocol, writeback))

    def _key(self, other):
        return f"{other}"

    def __repr__(self):
        filename = self._filename
        flag = self._flag
        protocol = self._protocol
        writeback = self._writeback
        return f"{type(self).__name__}({filename=}, {flag=}, {protocol=}, {writeback=})"

    def sync(self):
        return self._data.sync()
    
    def close(self):
        return self._data.close()


class Storage(collections.abc.MutableMapping):
    ROOT = pathlib.PurePosixPath('/')

    def __init__(self, backend, root=ROOT):
        self.root = root
        self.backend = backend

    def __fspath__(self):
        return str(self.root)
    
    def __truediv__(self, other):
        other = self._check_key(other)
        return type(self)(self.backend, other)

    @property
    def value(self):
        model =  self.backend.get(self._model_key(self.root.parent))
        value = self.backend.get(self.root)
        if model is None:
            return value
        return model(**value)
        
    
    @value.setter
    def value(self, other):
        model =  self.backend.get(self._model_key(self.root.parent))
        if model is None:
            self.backend[self.root] = other
        elif isinstance(other, model):
            self.backend[self.root] = dataclasses.asdict(other)
        else:
            raise ValueError

    @property
    def model(self):
        return self.backend.get(self._model_key(self.root))
    
    @model.setter
    def model(self, other):
        if dataclasses.is_dataclass(other) and isinstance(other, type):
            self.backend[self._model_key(self.root)] = other
        else:
            raise ValueError

    def _check_key(self, name):
        if not isinstance(name, (str, pathlib.PurePosixPath)):
            raise ValueError
        name = pathlib.PurePosixPath(name)
        if name.is_absolute() and name.is_relative_to(self.root):
            return name
        if name.is_absolute():
            raise KeyError
        return (self.root / name)

    def _model_key(self, name):
        name = pathlib.PurePosixPath(name)
        return f"!model:{name}"

    def __getitem__(self, name):
        return (self / name).value

    def __setitem__(self, name, value):
        name = self._check_key(name)
        (self / name).value = value

    def __delitem__(self, name):
        name = self._check_key(name)
        del self.backend[name]
    
    def __len__(self):
        return len(list(self.child_keys()))
    
    def __iter__(self):
        return self.child_keys()
    
    def child_keys(self, path=None, direct=False):
        path = path if path is not None else self.root
        path = pathlib.PurePosixPath(path)
        length = len(path.parts)
        for key in self.backend.keys():
            key = pathlib.PurePosixPath(key)
            if not key.is_relative_to(path):
                continue
            if direct and len(key.parts) > (length+1):
                continue
            yield key
    
    def child_values(self, path=None, direct=False):
        for key in self.child_keys(path, direct):
            yield self[key.relative_to(self.root)]
    
    def child_items(self, path=None, direct=False):
        for key in self.child_keys(path, direct):
            key = key.relative_to(self.root)
            yield key, self[key]

    def query(self, predicate=None):
        if predicate is None:
            return list(self.child_items(direct=True))
        if not callable(predicate):
            raise ValueError
        res = []
        for key, value in self.child_items(direct=True):
            value = value.value
            if predicate(value):
                res.append((key, value))
        return res
