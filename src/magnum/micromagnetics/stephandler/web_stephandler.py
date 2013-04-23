import BaseHTTPServer
import SocketServer

import magnum.solver.step_handler as stephandler
import magnum.logger as logger

import threading
#import json

import webbrowser


class MyHandler(BaseHTTPServer.BaseHTTPRequestHandler):

    def do_GET(self):
        siminfo = self.server.stephandler.get_new_siminfo()

        msg = "<html><body>\n"
        msg += "Simulation info:<br>"
        msg += "".join("<p>%s: %s</p>\n" % (k, v) for k, v in siminfo.__dict__.items())
        msg += "</body></html>\n"

        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf8")
        self.send_header("Content-Length", str(msg))
        #self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()

        self.wfile.write(msg)

    def log_message(self, fmt, *args):
        logger.info("Webservice - " + fmt % args)


class MyServer(BaseHTTPServer.HTTPServer, SocketServer.ThreadingMixIn):
    def __init__(self, server_address):
        super(MyServer, self).__init__(server_address, MyHandler)


class WebStepHandler(stephandler.StepHandler):

    class SimInfo(object):
        def __init__(self, state):
            self.t = state.t
            self.h = state.h
            self.deg_per_ns = state.deg_per_ns
            self.M_avg = state.M.average()

    def __init__(self, **kwargs):
        self.httpd = BaseHTTPServer.HTTPServer(("", 0), MyHandler)
        self.httpd.stephandler = self

        addr, port = self.httpd.server_address

        self.httpd_thread = threading.Thread(target=self.httpd.serve_forever)
        self.httpd_thread.daemon = True
        self.httpd_thread.start()

        self.siminfo = None
        self.siminfo_request = False

        logger.info("Starting webservice on port %s.", port)

        if kwargs.pop("open_browser", False):
            webbrowser.open_new_tab("http://localhost:%s" % port)

    def handle(self, state):
        if self.siminfo_request:
            self.siminfo = WebStepHandler.SimInfo(state)
            self.siminfo_request = False

    def done(self):
        self.httpd.shutdown()
        self.httpd_thread.join()

    def get_new_siminfo(self):
        self.siminfo_request = True
        while self.siminfo_request:
            pass
        return self.siminfo
