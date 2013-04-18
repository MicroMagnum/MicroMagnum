import subprocess
import xml.etree.ElementTree as etree

class NVidiaSmi(object):

    class GPU(object):
        def __repr__(self):
            return "NvidiaSmi.GPU(%s)" % self.__dict__

    def refresh(self):
        #cmd = "nvidia-smi -q -x".split()
        cmd = "cat nvidia_smi.out".split()
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            out, err = p.communicate()
            return self.parse(etree.fromstring(out))
        except OSError:
            return None

    @property
    def available(self):
        # Find available GPUs
        def check(gpu):
            if gpu.mode == 'Default':
                return True
            elif gpu.mode == 'Prohibited':
                return False
            else: # Exclusive_Thread, Exclusive_Process
                return gpu.num_processes == 0
        avail = [info[0] for info in enumerate(self.gpus) if check(info[1])]

        # Sort by memory load
        return sorted(avail, key=lambda id: self.gpus[id].mem_util, reverse=True)

    def parse(self, xml):
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

        return self.gpus

x = NVidiaSmi()
x.refresh()
print x.gpus[0]

#  	<gpu id="0000:14:00.0">
#  		<product_name>Tesla M2090</product_name>
#  		<display_mode>Disabled</display_mode>
#  		<persistence_mode>Disabled</persistence_mode>
#  		<driver_model>
#  			<current_dm>N/A</current_dm>
#  			<pending_dm>N/A</pending_dm>
#  		</driver_model>
#  		<serial>0324411088200</serial>
#  		<uuid>GPU-94c0c0f86546c136-ec582b09-95936242-25cd57e2-bdd5469ab0ec20002880d19b</uuid>
#  		<vbios_version>70.10.46.00.01</vbios_version>
#  		<inforom_version>
#  			<oem_object>1.1</oem_object>
#  			<ecc_object>2.0</ecc_object>
#  			<pwr_object>4.0</pwr_object>
#  		</inforom_version>
#  		<pci>
#  			<pci_bus>14</pci_bus>
#  			<pci_device>00</pci_device>
#  			<pci_domain>0000</pci_domain>
#  			<pci_device_id>109110DE</pci_device_id>
#  			<pci_bus_id>0000:14:00.0</pci_bus_id>
#  			<pci_sub_system_id>088710DE</pci_sub_system_id>
#  			<pci_gpu_link_info>
#  				<pcie_gen>
#  					<max_link_gen>2</max_link_gen>
#  					<current_link_gen>2</current_link_gen>
#  				</pcie_gen>
#  				<link_widths>
#  					<max_link_width>16x</max_link_width>
#  					<current_link_width>16x</current_link_width>
#  				</link_widths>
#  			</pci_gpu_link_info>
#  		</pci>
#  		<fan_speed>N/A</fan_speed>
#  		<performance_state>P0</performance_state>
#  		<memory_usage>
#  			<total>5375 MB</total>
#  			<used>2594 MB</used>
#  			<free>2781 MB</free>
#  		</memory_usage>
#  		<compute_mode>Exclusive_Thread</compute_mode>
#  		<utilization>
#  			<gpu_util>98 %</gpu_util>
#  			<memory_util>79 %</memory_util>
#  		</utilization>
#  		<ecc_mode>
#  			<current_ecc>Enabled</current_ecc>
#  			<pending_ecc>Enabled</pending_ecc>
#  		</ecc_mode>
#  		<temperature>
#  			<gpu_temp>N/A</gpu_temp>
#  		</temperature>
#  		<power_readings>
#  			<power_state>P0</power_state>
#  			<power_management>Supported</power_management>
#  			<power_draw>207.28 W</power_draw>
#  			<power_limit>225 W</power_limit>
#  		</power_readings>
#  		<clocks>
#  			<graphics_clock>650 MHz</graphics_clock>
#  			<sm_clock>1301 MHz</sm_clock>
#  			<mem_clock>1848 MHz</mem_clock>
#  		</clocks>
#  		<max_clocks>
#  			<graphics_clock>650 MHz</graphics_clock>
#  			<sm_clock>1301 MHz</sm_clock>
#  			<mem_clock>1848 MHz</mem_clock>
#  		</max_clocks>
#  		<compute_processes>
#  			<process_info>
#  				<pid>26931</pid>
#  				<process_name>/usr/bin/python</process_name>
#  				<used_memory>2577 MB</used_memory>
#  			</process_info>
#  		</compute_processes>
#  	</gpu>
