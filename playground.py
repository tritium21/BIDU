import collections
import functools
import pathlib
import re
import sys
import urllib.parse


class Template:
    TOK_RE = re.compile(r'({{.*?}}|{%.*?%})')
    
    def __init__(self, template_string):
        self.template_string = template_string

    @classmethod
    def from_file(cls, path):
        path = pathlib.Path(path)
        return cls(path.read_text(encoding='utf-8'))

    def split(self):
        return self.TOK_RE.split(self.template_string)

    def render(self, **context):
        return ''.join(self._render(**context))

    def _render(self, stream=None, **context):
        nested = False if stream is None else True
        stream = self.split() if stream is None else stream
        in_block = False
        for index, fragment in enumerate(stream):
            if fragment[:2] not in {'{{', '{%'}:
                if in_block:
                    continue
                yield fragment
            elif fragment[:2] == '{{':
                if in_block:
                    continue
                name = fragment[2:-2].strip()
                yield str(eval(name, {'__builtins__': None}, context.copy()))
            elif fragment[:2] == '{%':
                contents = fragment[2:-2].strip()
                match = re.match('for (.*?) in (.*)', contents)
                if match:
                    if in_block:
                        continue
                    in_block = True
                    var, obj = match.groups()
                    for it in eval(obj, {'__builtins__': None}, context.copy()):
                        ctx = {var: it}
                        yield from self._render(stream=stream[index + 1:], **ctx)
                elif contents == 'endfor':
                    if not in_block:
                        break
                    in_block = False

class Request(collections.Mapping):
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
            if ((scheme == 'https' and port != '443') or
                (scheme == 'http' and port != '80')):
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
        if isinstance(self._headers, (dict, collections.Mapping)):
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
        if not isinstance(other, (str,Request,Rule)):
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

    def route(self, function_or_route, *, route=None, endpoint=None):
        if not callable(function_or_route):
            return functools.partial(self.route, route=function_or_route)
        rule = Rule(route, function_or_route, endpoint)
        self.routes.append(rule)
        self.endpoints[rule.endpoint].append(rule)
        return function_or_route

    def dispatch(self, request):
        for route in self.routes:
            if request != route:
                continue
            main, redirect = route.match(request.path_info)
            if main:
                return route(request)
            kwargs = redirect.groupdict()
            print(kwargs)
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
    import pprint
    import os
    from wsgiref.simple_server import make_server
    HOST = os.environ.get('SERVER_HOST', '')
    PORT = int(os.environ.get('SERVER_PORT', 8000))

    application = Application()
    
    @application.route('/')
    def root(request):
        env = pprint.pformat(request.environ)
        query = pprint.pformat(request.query)
        method = request.method
        path_info = request.path_info
        url_for = application.url_for('root')
        body = f'''
This: {request.this}

url_for test: {url_for}

Path: {path_info}

Query: {query}

Method: {method}

{env}'''
        return Response(body, content_type='text/plain')

    @application.route('/hello/<bar>/')
    @application.route('/hello/')
    def hello(request, bar='World'):
        template = Template.from_file('template.html')
        title = f'Hello, {bar}'
        body = "I am the very model of a modern major general!"
        return Response(template.render(title=title, body=body))

    @application.route('/foo/<count:int>')
    def foo(request, count):
        return Response('Foo!\n' * count, content_type='text/plain')

    with make_server(HOST, PORT, application) as httpd:
        httpd.serve_forever()
