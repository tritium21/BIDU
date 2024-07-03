import datetime
import getpass
import pathlib

from bidu import Application, Response, Storage, MemoryBackend

application = Application(template_dir=pathlib.Path(__file__).parent)
application["site_name"] = "Demo Site"

storage = Storage(MemoryBackend())
storage['count'] = 0

@application.router(method='GET', route='/')
def root(request):
    return Response('', status=302, headers={'location': request.url_for('hello', bar=getpass.getuser())})

@application.get(route='/hello/<bar>/')
@application.get(route='/hello/')
def hello(request, bar='World'):
    now = datetime.datetime.now().isoformat()
    then = t.value if (t := request.cookies.get("now")) else "never"
    title = f'Hello, {bar}'
    body = "I am the very model of a modern major general!"
    items = request.query.get("items", "This is a list of strings".split())
    count = (storage / "count").value
    count += 1
    (storage / "count").value = count
    resp = Response(request.application.template("template.tmpl", title=title, item=body, items=items, then=then, count=count))
    resp.cookies['now'] = str(now)
    resp.cookies['now']['max-age'] = 300
    return resp

if __name__ == "__main__":
    import os
    import wsgiref.simple_server

    HOST = os.environ.get('SERVER_HOST', '')
    PORT = int(os.environ.get('SERVER_PORT', 8000))
    with wsgiref.simple_server.make_server(HOST, PORT, application) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
