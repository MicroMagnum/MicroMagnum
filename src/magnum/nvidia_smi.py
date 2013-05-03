import subprocess
import xml.etree.ElementTree as ElementTree

import magnum.logger as logger


def run(cmd="nvidia-smi -q -x"):
    """
    Runs nvidia-smi and returns a list of reported GPUs
    """
    try:
        p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        out, err = p.communicate()
        return _parse(ElementTree.fromstring(out))
    except ElementTree.ParseError:
        logger.error("nvidia-smi output could not be parsed.")
        raise
    except OSError:
        logger.error("nvidia-smi tool could not be executed.")
        raise


def _parse(xml):
    gpus = []

    for gpu_id, gpu_xml in enumerate(xml.findall('gpu')):
        gpu = dict(id=gpu_id, xml=gpu_xml)

        # Identity
        gpu['name'] = gpu_xml.find('product_name').text.strip()

        # GPU load
        gpu['gpu_util'] = float(gpu_xml.find('utilization/gpu_util').text.replace("%","")) / 100.0
        gpu['mem_util'] = float(gpu_xml.find('utilization/memory_util').text.replace("%","")) / 100.0

        # Memory usage
        gpu['mem_total'] = gpu_xml.find('memory_usage/total').text.strip()
        gpu['mem_used']  = gpu_xml.find('memory_usage/used').text.strip()
        gpu['mem_free']  = gpu_xml.find('memory_usage/free').text.strip()

        # Compute mode
        gpu['mode'] = gpu_xml.find('compute_mode').text.strip()

        # Number of processes running on this GPU
        gpu['num_processes'] = len(gpu_xml.findall('compute_processes/process_info'))

        gpus.append(gpu)

    return gpus


def is_available(gpu):
    """
    Check if GPU is available.
    """

    if gpu['mode'] == 'Default':
        return True
    elif gpu['mode'] == 'Prohibited':
        return False
    else: # Exclusive_Thread, Exclusive_Process
        return gpu['num_processes'] == 0 # available only if no-one is using the GPU.


def available(gpus, rank_by="mem_util", reverse=True):
    """
    Return GPUs reported by nvidia-smi that are available. Available
    means: Compute mode is either Default (allowing more than one process
    per GPU) or the GPU is currently unused (the number of processes on
    that GPU is zero). By default, the returned list is sorted by the
    GPU load, with the least loaded cards first.
    """
    # 1. Filter out unavailable GPUs
    avail = filter(is_available, gpus)
    # 2. Sort by memory load, if requested.
    if rank_by:
        avail = sorted(avail, key=lambda gpu: gpu[rank_by], reverse=reverse)
    return avail


def run_with_test_output():
    return _parse(ElementTree.fromstring(
        """<?xml version="1.0" ?>
        <!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v2.dtd">
        <nvidia_smi_log>
        	<timestamp>Thu Apr 18 15:50:36 2013</timestamp>
        	<driver_version>285.05.32</driver_version>
        	<attached_gpus>2</attached_gpus>
        	<gpu id="0000:14:00.0">
        		<product_name>Tesla M2090</product_name>
        		<display_mode>Disabled</display_mode>
        		<persistence_mode>Disabled</persistence_mode>
        		<driver_model>
        			<current_dm>N/A</current_dm>
        			<pending_dm>N/A</pending_dm>
        		</driver_model>
        		<serial>0324411088200</serial>
        		<uuid>GPU-94c0c0f86546c136-ec582b09-95936242-25cd57e2-bdd5469ab0ec20002880d19b</uuid>
        		<vbios_version>70.10.46.00.01</vbios_version>
        		<inforom_version>
        			<oem_object>1.1</oem_object>
        			<ecc_object>2.0</ecc_object>
        			<pwr_object>4.0</pwr_object>
        		</inforom_version>
        		<pci>
        			<pci_bus>14</pci_bus>
        			<pci_device>00</pci_device>
        			<pci_domain>0000</pci_domain>
        			<pci_device_id>109110DE</pci_device_id>
        			<pci_bus_id>0000:14:00.0</pci_bus_id>
        			<pci_sub_system_id>088710DE</pci_sub_system_id>
        			<pci_gpu_link_info>
        				<pcie_gen>
        					<max_link_gen>2</max_link_gen>
        					<current_link_gen>2</current_link_gen>
        				</pcie_gen>
        				<link_widths>
        					<max_link_width>16x</max_link_width>
        					<current_link_width>16x</current_link_width>
        				</link_widths>
        			</pci_gpu_link_info>
        		</pci>
        		<fan_speed>N/A</fan_speed>
        		<performance_state>P0</performance_state>
        		<memory_usage>
        			<total>5375 MB</total>
        			<used>2594 MB</used>
        			<free>2781 MB</free>
        		</memory_usage>
        		<compute_mode>Exclusive_Thread</compute_mode>
        		<utilization>
        			<gpu_util>98 %</gpu_util>
        			<memory_util>79 %</memory_util>
        		</utilization>
        		<ecc_mode>
        			<current_ecc>Enabled</current_ecc>
        			<pending_ecc>Enabled</pending_ecc>
        		</ecc_mode>
        		<ecc_errors>
        			<volatile>
        				<single_bit>
        					<device_memory>0</device_memory>
        					<register_file>0</register_file>
        					<l1_cache>0</l1_cache>
        					<l2_cache>0</l2_cache>
        					<total>0</total>
        				</single_bit>
        				<double_bit>
        					<device_memory>0</device_memory>
        					<register_file>0</register_file>
        					<l1_cache>0</l1_cache>
        					<l2_cache>0</l2_cache>
        					<total>0</total>
        				</double_bit>
        			</volatile>
        			<aggregate>
        				<single_bit>
        					<device_memory>0</device_memory>
        					<register_file>0</register_file>
        					<l1_cache>0</l1_cache>
        					<l2_cache>0</l2_cache>
        					<total>0</total>
        				</single_bit>
        				<double_bit>
        					<device_memory>0</device_memory>
        					<register_file>0</register_file>
        					<l1_cache>0</l1_cache>
        					<l2_cache>0</l2_cache>
        					<total>0</total>
        				</double_bit>
        			</aggregate>
        		</ecc_errors>
        		<temperature>
        			<gpu_temp>N/A</gpu_temp>
        		</temperature>
        		<power_readings>
        			<power_state>P0</power_state>
        			<power_management>Supported</power_management>
        			<power_draw>207.28 W</power_draw>
        			<power_limit>225 W</power_limit>
        		</power_readings>
        		<clocks>
        			<graphics_clock>650 MHz</graphics_clock>
        			<sm_clock>1301 MHz</sm_clock>
        			<mem_clock>1848 MHz</mem_clock>
        		</clocks>
        		<max_clocks>
        			<graphics_clock>650 MHz</graphics_clock>
        			<sm_clock>1301 MHz</sm_clock>
        			<mem_clock>1848 MHz</mem_clock>
        		</max_clocks>
        		<compute_processes>
        			<process_info>
        				<pid>26931</pid>
        				<process_name>/usr/bin/python</process_name>
        				<used_memory>2577 MB</used_memory>
        			</process_info>
        			<process_info>
        				<pid>26931</pid>
        				<process_name>/usr/bin/python</process_name>
        				<used_memory>2577 MB</used_memory>
        			</process_info>
        		</compute_processes>
        	</gpu>

        	<gpu id="0000:15:00.0">
        		<product_name>Tesla M2090</product_name>
        		<display_mode>Disabled</display_mode>
        		<persistence_mode>Disabled</persistence_mode>
        		<driver_model>
        			<current_dm>N/A</current_dm>
        			<pending_dm>N/A</pending_dm>
        		</driver_model>
        		<serial>0324111049013</serial>
        		<uuid>GPU-994ae25fdc2f6444-0b805696-72581ff7-ae34b370-821d8fbbf18c038a8c5c33ca</uuid>
        		<vbios_version>70.10.46.00.01</vbios_version>
        		<inforom_version>
        			<oem_object>1.1</oem_object>
        			<ecc_object>2.0</ecc_object>
        			<pwr_object>4.0</pwr_object>
        		</inforom_version>
        		<pci>
        			<pci_bus>15</pci_bus>
        			<pci_device>00</pci_device>
        			<pci_domain>0000</pci_domain>
        			<pci_device_id>109110DE</pci_device_id>
        			<pci_bus_id>0000:15:00.0</pci_bus_id>
        			<pci_sub_system_id>088710DE</pci_sub_system_id>
        			<pci_gpu_link_info>
        				<pcie_gen>
        					<max_link_gen>2</max_link_gen>
        					<current_link_gen>1</current_link_gen>
        				</pcie_gen>
        				<link_widths>
        					<max_link_width>16x</max_link_width>
        					<current_link_width>16x</current_link_width>
        				</link_widths>
        			</pci_gpu_link_info>
        		</pci>
        		<fan_speed>N/A</fan_speed>
        		<performance_state>P12</performance_state>
        		<memory_usage>
        			<total>5375 MB</total>
        			<used>10 MB</used>
        			<free>5364 MB</free>
        		</memory_usage>
        		<compute_mode>Exclusive_Thread</compute_mode>
        		<utilization>
        			<gpu_util>0 %</gpu_util>
        			<memory_util>0 %</memory_util>
        		</utilization>
        		<ecc_mode>
        			<current_ecc>Enabled</current_ecc>
        			<pending_ecc>Enabled</pending_ecc>
        		</ecc_mode>
        		<ecc_errors>
        			<volatile>
        				<single_bit>
        					<device_memory>0</device_memory>
        					<register_file>0</register_file>
        					<l1_cache>0</l1_cache>
        					<l2_cache>0</l2_cache>
        					<total>0</total>
        				</single_bit>
        				<double_bit>
        					<device_memory>0</device_memory>
        					<register_file>0</register_file>
        					<l1_cache>0</l1_cache>
        					<l2_cache>0</l2_cache>
        					<total>0</total>
        				</double_bit>
        			</volatile>
        			<aggregate>
        				<single_bit>
        					<device_memory>0</device_memory>
        					<register_file>0</register_file>
        					<l1_cache>0</l1_cache>
        					<l2_cache>0</l2_cache>
        					<total>0</total>
        				</single_bit>
        				<double_bit>
        					<device_memory>0</device_memory>
        					<register_file>0</register_file>
        					<l1_cache>0</l1_cache>
        					<l2_cache>0</l2_cache>
        					<total>0</total>
        				</double_bit>
        			</aggregate>
        		</ecc_errors>
        		<temperature>
        			<gpu_temp>N/A</gpu_temp>
        		</temperature>
        		<power_readings>
        			<power_state>P12</power_state>
        			<power_management>Supported</power_management>
        			<power_draw>32.03 W</power_draw>
        			<power_limit>225 W</power_limit>
        		</power_readings>
        		<clocks>
        			<graphics_clock>50 MHz</graphics_clock>
        			<sm_clock>101 MHz</sm_clock>
        			<mem_clock>135 MHz</mem_clock>
        		</clocks>
        		<max_clocks>
        			<graphics_clock>650 MHz</graphics_clock>
        			<sm_clock>1301 MHz</sm_clock>
        			<mem_clock>1848 MHz</mem_clock>
        		</max_clocks>
        		<compute_processes>
        		</compute_processes>
        	</gpu>
        </nvidia_smi_log>
        """
    ))
