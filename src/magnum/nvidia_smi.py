import subprocess
import xml.etree.ElementTree as ElementTree

import magnum.logger as logger


class NVidiaSmi(object):

    class GPU(object):
        def __repr__(self):
            return "NvidiaSmi.GPU(%s)" % self.__dict__

    def __init__(self):
        self.gpus = []

    def refresh(self):
        cmd = "nvidia-smi -q -x".split()
        #cmd = "cat nvidia_smi.out".split()
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            out, err = p.communicate()
            return self._parse(ElementTree.fromstring(out))
        except ElementTree.ParseError:
            logger.error("NVidiaSmi: nvidia-smi output could not be parsed.")
            raise
        except OSError:
            logger.error("NVidiaSmi: nvidia-smi tool could not be executed.")
            raise

    @property
    def all(self):
        """
        Returns info about all GPUs reported by the nvidia-smi tool.
        """
        return self.gpus

    @property
    def available(self):
        """
        Return GPUs reported by nvidia-smi that are available. Available
        means: Compute mode is either Default (allowing more than one
        process per GPU) or the GPU is currently unused (the number of
        processes on that GPU is zero). The returned list is sorted by
        the GPU load, with the least loaded cards first.
        """
        def check(gpu):
            # Check if GPU is available
            if gpu.mode == 'Default':
                return True
            elif gpu.mode == 'Prohibited':
                return False
            else: # Exclusive_Thread, Exclusive_Process
                return gpu.num_processes == 0 # available only if no-one is using the GPU.
        avail = [info[0] for info in enumerate(self.gpus) if check(info[1])]

        # Sort by memory load
        return sorted(avail, key=lambda id: self.gpus[id].mem_util, reverse=True)

    def _parse(self, xml):
        self.xml = xml
        self.gpus = []

        for id, gpu_xml in enumerate(xml.findall('gpu')):
            gpu = NVidiaSmi.GPU()
            gpu.id = id
            gpu.xml = gpu_xml

            # GPU load
            gpu.gpu_util = float(gpu.xml.find('utilization/gpu_util').text.replace("%","")) / 100.0
            gpu.mem_util = float(gpu.xml.find('utilization/memory_util').text.replace("%","")) / 100.0

            # Memory usage
            gpu.mem_total = gpu.xml.find('memory_usage/total').text
            gpu.mem_used  = gpu.xml.find('memory_usage/used').text
            gpu.mem_free  = gpu.xml.find('memory_usage/free').text

            # Compute mode
            gpu.mode = gpu.xml.find('compute_mode').text.strip()

            # Number of processes running on this GPU
            gpu.num_processes = len(gpu.xml.findall('compute_processes/process_info'))

            self.gpus.append(gpu)
