from magnum.external.bottle import Bottle, static_file

import magnum.solver.step_handler as stephandler
import magnum.logger as logger

import threading
import webbrowser

from wsgiref.simple_server import make_server

import sys
import os

def make_bottle_app(stephandler):

    app = Bottle()

    @app.route("/")
    def get():
        siminfo = stephandler.get_siminfo(refresh=True)
        return "<ul>" + "".join("<li>%s: %s</li>" % (k, v) for k, v in siminfo.data.items()) + "</ul>"

    @app.route("/json")
    def get():
        siminfo = stephandler.get_siminfo(refresh=True)
        return siminfo.data

    static_path = os.path.dirname(sys.modules[__name__].__file__) + "/static"

    @app.route('/static/<filename>')
    def serve_static(filename):
        return static_file(filename, root=static_path)

    @app.route('/favicon.ico')
    def favicon():
        return static_file('favicon.ico', root=static_path)

    return app

class SimInfo(object):
    IDS = []
    IDS += ["t", "h", "deg_per_ns", "step"]
    IDS += ["E_tot", "E_exch", "E_stray", "E_aniso", "E_ext"]

    def __init__(self, state):
        self.data = {}
        for id in self.IDS:
            if hasattr(state, id):
                self.data[id] = getattr(state, id)
        self.data["M_avg"] = state.M.average()

class WebStepHandler(stephandler.StepHandler):

    def __init__(self, **kwargs):
        app = make_bottle_app(self)
        self.httpd = make_server('', 0, app)

        self.httpd_thread = threading.Thread(target=self.httpd.serve_forever)
        self.httpd_thread.daemon = True
        self.httpd_thread.start()

        self.siminfo = None
        self.siminfo_request = False

        addr, port = self.httpd.server_address
        logger.info("Starting webservice on port %s.", port)

        if kwargs.pop("open_browser", False):
            webbrowser.open_new_tab("http://localhost:%s" % port)

    def handle(self, state):
        if self.siminfo_request:
            self.siminfo = SimInfo(state)
            self.siminfo_request = False

    def done(self):
        self.httpd.shutdown()
        self.httpd_thread.join()

    def get_siminfo(self, refresh=False):
        if refresh or not self.siminfo:
            self.siminfo_request = True
            while self.siminfo_request:
                pass
        return self.siminfo
