from enum import Enum

from attr import attrs, attrib
import pyspark.sql.functions as F
from pyspark.sql.types import StringType


class DeviceUse(str, Enum):
    Speaker = 'speaker'
    Show = 'show'
    Plug = 'plug'
    Earphone = 'earphone'
    GlassFrame = 'glass_frame'
    Camera = 'camera'
    Auto = 'auto'
    TV = 'tv'


@attrs(slots=True)
class DeviceType:
    ID = attrib(type=str)
    CodeName = attrib(type=str)
    ExternalName = attrib(type=str)
    ReleaseYear = attrib(type=int)
    HasScreen = attrib(type=bool, default=False)
    Use = attrib(type=DeviceUse, default=DeviceUse.Speaker)
    System = attrib(type=str, default=None)


def get_known_no_screen_non_tv_device_types():
    from boba_python_utils.production_utils.common._constants.device_type import DEVICE_TYPE_INFO
    return {
        device_type.ID
        for device_type in DEVICE_TYPE_INFO.values()
        if not (device_type.HasScreen or device_type.Use == DeviceUse.TV)
    }


def get_echo_show_device_types():
    from boba_python_utils.production_utils.common._constants.device_type import DEVICE_TYPE_INFO
    return {
        device_type.ID
        for device_type in DEVICE_TYPE_INFO.values()
        if device_type.Use == DeviceUse.Show
    }


def get_tv_device_types():
    from boba_python_utils.production_utils.common._constants.device_type import DEVICE_TYPE_INFO
    return {
        device_type.ID
        for device_type in DEVICE_TYPE_INFO.values()
        if device_type.Use == DeviceUse.TV
    }

def get_device_external_name(device_type):
    from boba_python_utils.production_utils.common._constants.device_type import DEVICE_TYPE_INFO
    if device_type in DEVICE_TYPE_INFO:
        return DEVICE_TYPE_INFO[device_type].ExternalName


get_device_external_name_udf = F.udf(get_device_external_name, returnType=StringType())

