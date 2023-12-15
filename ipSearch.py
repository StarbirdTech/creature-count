import threading
from zeroconf import ServiceBrowser, Zeroconf


class MyListener:
    def __init__(self, target_name, found_condition):
        self.target_name = target_name
        self.found_condition = found_condition
        self.ip_address = None

    def add_service(self, zeroconf, type, name):
        if name.startswith(self.target_name):
            info = zeroconf.get_service_info(type, name)
            if info:
                addresses = [
                    ".".join(map(str, addr)) for addr in info.parsed_addresses()
                ]
                if addresses:
                    with self.found_condition:
                        self.ip_address = addresses[0]
                        self.found_condition.notify()

    def update_service(self, zeroconf, type, name, state_change):
        # This method is required, but you can leave it empty if you don't need to handle updates
        pass

    def remove_service(self, zeroconf, type, name):
        # Handle the service removal if necessary
        pass


def get_ip(target_name):
    found_condition = threading.Condition()
    listener = MyListener(target_name, found_condition)
    zeroconf = Zeroconf()
    browser = ServiceBrowser(zeroconf, "_http._tcp.local.", listener)
    print("Searching for {}...".format(target_name))
    with found_condition:
        found_condition.wait()
        ip_address = listener.ip_address

    zeroconf.close()

    ip_address = ip_address.replace("...", ",")
    ip_address = ip_address.replace(".", "")
    ip_address = ip_address.replace(",", ".")

    return ip_address
