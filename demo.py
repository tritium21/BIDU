
import datetime
import getpass
import os
from wsgiref.simple_server import make_server

from bidu import Application, Response

HOST = os.environ.get('SERVER_HOST', '')
PORT = int(os.environ.get('SERVER_PORT', 8000))

application = Application()

@application.router(method='GET', route='/')
def root(request):
    return Response('', status=302, headers={'location': application.url_for('hello', bar=getpass.getuser())})

@application.get(route='/hello/<bar>/')
@application.get(route='/hello/')
def hello(request, bar='World'):
    now = datetime.datetime.now().isoformat()
    then = t.value if (t := request.cookies.get("now")) else "never"
    title = f'Hello, {bar}'
    body = "I am the very model of a modern major general!"
    items = request.query.get("items", "This is a list of strings".split())
    resp = Response(application.templates.render("demo.html.tmpl", title=title, body=body, items=items, then=then))
    resp.cookies['now'] = str(now)
    resp.cookies['now']['max-age'] = 300
    return resp

if __name__ == "__main__":
    with make_server(HOST, PORT, application) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
