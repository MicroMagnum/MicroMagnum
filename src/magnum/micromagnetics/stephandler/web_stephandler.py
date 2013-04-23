import BaseHTTPServer
import SocketServer

import magnum.solver.step_handler as stephandler
import magnum.logger as logger

import threading
import webbrowser
#import json


class MyHandler(BaseHTTPServer.BaseHTTPRequestHandler):

    def do_GET(self):
        siminfo = self.server.stephandler.get_new_siminfo()

        msg = "<html><body>\n"
        msg += "Simulation info:<br>"
        msg += "".join("<p>%s: %s</p>\n" % (k, v) for k, v in siminfo.data)
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

class SimInfo(object):
    IDS = []
    IDS += ["t", "h", "deg_per_ns"]
    IDS += ["E_tot", "E_exch", "E_stray", "E_aniso", "E_ext"]

    def __init__(self, state):
        self.data = []
        for id in SimInfo.IDS:
            if hasattr(state, id):
                self.data.append((id, getattr(state, id)))
        self.data.append(("M_avg", state.M.average()))
        self.data = sorted(self.data, key=lambda x: x[0])

class WebStepHandler(stephandler.StepHandler):

    def __init__(self, **kwargs):
        self.httpd = BaseHTTPServer.HTTPServer(("", 0), MyHandler)
        self.httpd.stephandler = self

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

    def get_new_siminfo(self):
        # TODO: Use Condition variable or threading.Event
        self.siminfo_request = True
        while self.siminfo_request:
            pass
        return self.siminfo
