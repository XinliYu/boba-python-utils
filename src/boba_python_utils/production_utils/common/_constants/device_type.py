from enum import Enum

from attr import attrs, attrib

from boba_python_utils.production_utils.common.device_type import DeviceType, DeviceUse

DEVICE_TYPE_INFO = {
    # region Echo Speakers
    'AB72C64C86AW2': DeviceType(
        ID='AB72C64C86AW2',
        CodeName='Doppler',
        ExternalName='Echo',
        ReleaseYear=2014
    ),
    'A7WXQPH584YP': DeviceType(
        ID='A7WXQPH584YP',
        CodeName='Radar',
        ExternalName='Echo (v2)',
        ReleaseYear=2017,
        System='Fire OS 5'
    ),
    'A38949IHXHRQ5P': DeviceType(
        ID='A38949IHXHRQ5P',
        CodeName='Fox',
        ExternalName='Echo Tap',
        ReleaseYear=2016
    ),
    'AKNO1N0KSFN8L': DeviceType(
        ID='AKNO1N0KSFN8L',
        CodeName='Pancake',
        ExternalName='Echo Dot',
        ReleaseYear=2015
    ),
    'A3S5BH2HU6VAYF': DeviceType(
        ID='A3S5BH2HU6VAYF',
        CodeName='Biscuit',
        ExternalName='Echo Dot (v2)',
        ReleaseYear=2017,
        System='Fire OS 5'
    ),
    'A32DOYMUN6DTXA': DeviceType(
        ID='A32DOYMUN6DTXA',
        CodeName='Donut',
        ExternalName='Echo Dot (v3)',
        ReleaseYear=2018,
        System='Fire OS 6'
    ),
    'A1RABVCI4QCIKC': DeviceType(
        ID='A1RABVCI4QCIKC',
        CodeName='Crumpet',
        ExternalName='Echo Dot Gen 3',
        ReleaseYear=2019,
        System='Puffin'
    ),
    'A30YDR2MK8HMRV': DeviceType(
        ID='A30YDR2MK8HMRV',
        CodeName='Doebrite',
        ExternalName='Echo Dot Gen 3 with Clock',
        ReleaseYear=2019,
        System='Puffin'
    ),
    'A2U21SRK4QGSE1': DeviceType(
        ID='A2U21SRK4QGSE1',
        CodeName='Brownie',
        ExternalName='Echo Dot Gen 4',
        ReleaseYear=2020,
        System='Puffin'
    ),
    'A2H4LV5GIZ1JFT': DeviceType(
        ID='A2H4LV5GIZ1JFT',
        CodeName='Ganache',
        ExternalName='Echo Dot (4th Gen) with Clock',
        ReleaseYear=2020,
        System='Puffin'
    ),
    'A2DS1Q2TPDJ48U': DeviceType(
        ID='A2DS1Q2TPDJ48U',
        CodeName='Cheesecake',
        ExternalName='Echo Dot (5th Gen) with clock',
        ReleaseYear=2022,
        System='Puffin'
    ),
    'A1JJ0KFC4ZPNJ3': DeviceType(
        ID='A1JJ0KFC4ZPNJ3',
        CodeName='Cupcake',
        ExternalName='Echo Input',
        ReleaseYear=2018,
        System='Fire OS 6'
    ),
    'A2M35JJZWCQOMZ': DeviceType(
        ID='A2M35JJZWCQOMZ',
        CodeName='Sonar',
        ExternalName='Echo Plus',
        ReleaseYear=2017,
        System='Fire OS 5'
    ),
    'A18O6U1UQFJ0XK': DeviceType(
        ID='A18O6U1UQFJ0XK',
        CodeName='Lidar',
        ExternalName='Echo Plus (v2)',
        ReleaseYear=2018,
        System='Fire OS 6'
    ),
    'A3FX4UWTP28V1P': DeviceType(
        ID='A3FX4UWTP28V1P',
        CodeName='Pascal',
        ExternalName='Echo (v3)',
        ReleaseYear=2019,
        System='Puffin'
    ),
    'A3RMGO6LYLH7YN': DeviceType(
        ID='A3RMGO6LYLH7YN',
        CodeName='Laser',
        ExternalName='Echo (v4)',
        ReleaseYear=2020,
        System='Puffin'
    ),
    'A3RBAYBE7VM004': DeviceType(
        ID='A3RBAYBE7VM004',
        CodeName='Octave',
        ExternalName='Echo Studio',
        ReleaseYear=2019,
        System='Fire OS 6'
    ),
    # endregion,

    # region Echo Show
    'A1NL4BVLQ4L3N3': DeviceType(
        ID='A1NL4BVLQ4L3N3',
        CodeName='Knight',
        ExternalName='Echo Show',
        HasScreen=True,
        Use=DeviceUse.Show,
        ReleaseYear=2017,
        System='Fire OS 5'
    ),
    'A10A33FOX2NUBK': DeviceType(
        ID='A10A33FOX2NUBK',
        CodeName='Rook',
        ExternalName='Echo Spot',
        HasScreen=True,
        ReleaseYear=2017,
        System='Fire OS 5'
    ),
    'AWZZ5CVHX2CD': DeviceType(
        ID='AWZZ5CVHX2CD',
        CodeName='Bishop',
        ExternalName='Echo Show (v2)',
        Use=DeviceUse.Show,
        HasScreen=True,
        ReleaseYear=2018,
        System='Fire OS 5'
    ),
    'A4ZP7ZC4PI6TO': DeviceType(
        ID='A4ZP7ZC4PI6TO',
        CodeName='Checkers',
        ExternalName='Echo Show 5',
        Use=DeviceUse.Show,
        HasScreen=True,
        ReleaseYear=2019,
        System='Fire OS 6'
    ),
    'A1Z88NGR2BK6A2': DeviceType(
        ID='A1Z88NGR2BK6A2',
        CodeName='Crown',
        ExternalName='Echo Show 8',
        Use=DeviceUse.Show,
        HasScreen=True,
        ReleaseYear=2019,
        System='Fire OS 6'
    ),
    'AIPK7MM90V7TB': DeviceType(
        ID='AIPK7MM90V7TB',
        CodeName='Theia',
        ExternalName='Echo Show 10',
        Use=DeviceUse.Show,
        HasScreen=True,
        ReleaseYear=2021,
        System='Fire OS 7'
    ),
    'A15996VY63BQ2D': DeviceType(
        ID='A15996VY63BQ2D',
        CodeName='Athena',
        ExternalName='Echo Show 8 (Gen 2)',
        Use=DeviceUse.Show,
        HasScreen=True,
        ReleaseYear=2021,
        System='Fire OS 7'
    ),
    'A1EIANJ7PNB0Q7': DeviceType(
        ID='A1EIANJ7PNB0Q7',
        CodeName='Hoya',
        ExternalName='Echo Show 15',
        HasScreen=True,
        ReleaseYear=2021,
        System='Fire OS 7'
    ),
    'A1XWJRHALS1REP': DeviceType(
        ID='A1XWJRHALS1REP',
        CodeName='Cronos',
        ExternalName='Echo Show 5 (Gen 2)',
        Use=DeviceUse.Show,
        HasScreen=True,
        ReleaseYear=2021,
        System='Fire OS 6'
    ),
    # endregion

    # region Echo Misc
    'A3VRME03NAXFUB': DeviceType(
        ID='A3VRME03NAXFUB',
        CodeName='Croissant',
        ExternalName='Echo Flex',
        Use=DeviceUse.Plug,
        ReleaseYear=2019,
        System='Puffin'
    ),
    'A16MZVIFVHX6P6': DeviceType(
        ID='A16MZVIFVHX6P6',
        CodeName='Puget',
        ExternalName='Echo Buds',
        Use=DeviceUse.Earphone,
        ReleaseYear=2019
    ),
    'A15QWUTQ6FSMYX': DeviceType(
        ID='A15QWUTQ6FSMYX',
        CodeName='Powell',
        ExternalName='Echo Buds',
        Use=DeviceUse.Earphone,
        ReleaseYear=2021
    ),
    'A3IYPH06PH1HRA': DeviceType(
        ID='A3IYPH06PH1HRA',
        CodeName='Zebra',
        ExternalName='Echo Frames',
        Use=DeviceUse.GlassFrame,
        ReleaseYear=2021
    ),
    'A1ORT4KZ23OY88': DeviceType(
        ID='A1ORT4KZ23OY88',
        CodeName='Hendrix',
        ExternalName='Echo Look',
        Use=DeviceUse.Camera,
        ReleaseYear=2017,
        System='FireOS 5'
    ),
    # endregion

    # region FireTV
    'A8MCGN45KMHDH': DeviceType(
        ID='A8MCGN45KMHDH',
        CodeName='Kaine/Kayla',
        ExternalName='Fire TV',
        Use=DeviceUse.TV,
        HasScreen=True,
        ReleaseYear=2021,
        System='FOS7'
    ),
    'A2E0SNTXJVT7WK': DeviceType(
        ID='A2E0SNTXJVT7WK',
        CodeName='Bueller',
        ExternalName='First Fire TV device',
        Use=DeviceUse.TV,
        ReleaseYear=2014
    ),
    'A12GXV8XMS007S': DeviceType(
        ID='A12GXV8XMS007S',
        CodeName='Sloane',
        ExternalName='Second Fire TV device (4K UHD)',
        Use=DeviceUse.TV,
        ReleaseYear=2015
    ),
    'A3HF4YRA2L7XGC': DeviceType(
        ID='A3HF4YRA2L7XGC',
        CodeName='Stark',
        ExternalName='Fire TV Cube',
        Use=DeviceUse.TV,
        ReleaseYear=2018,
        System='FireOS 6'
    ),
    'A2JKHJ0PX4J3L3': DeviceType(
        ID='A2JKHJ0PX4J3L3',
        CodeName='Raven',
        ExternalName='Fire TV Cube',
        Use=DeviceUse.TV,
        ReleaseYear=2019
    ),
    'A2LWARUGJLBYEW': DeviceType(
        ID='A2LWARUGJLBYEW',
        CodeName='Tank',
        ExternalName='Fire TV Stick with Voice Remote',
        Use=DeviceUse.TV,
        ReleaseYear=2016
    ),
    'A2GFL5ZMWNE0PX': DeviceType(
        ID='A2GFL5ZMWNE0PX',
        CodeName='Needle',
        ExternalName='Fire TV Stick 4K with Alexa Voice Remote (1st Gen)',
        Use=DeviceUse.TV,
        ReleaseYear=2018
    ),
    'AKPGW064GI9HE': DeviceType(
        ID='AKPGW064GI9HE',
        CodeName='Mantis',
        ExternalName='Fire TV Stick 4K with Alexa Voice Remote (2nd Gen, released 2019)',
        Use=DeviceUse.TV,
        ReleaseYear=2019
    ),
    'A35P7LGLKFSYUP': DeviceType(
        ID='A35P7LGLKFSYUP',
        CodeName='Bane',
        ExternalName='Alexa Voice Remote 1st Gen',
        Use=DeviceUse.TV,
        ReleaseYear=2014
    ),
    'A2RW5QTD9VSRD8': DeviceType(
        ID='A2RW5QTD9VSRD8',
        CodeName='Lindbergh',
        ExternalName='Alexa Voice Remote 2nd Gen',
        Use=DeviceUse.TV,
        ReleaseYear=2016
    ),
    'AGHZIK8D6X7QR': DeviceType(
        ID='AGHZIK8D6X7QR',
        CodeName='Rita/Margo',
        ExternalName='Amazon TV w/ FireTV Built-In',
        HasScreen=True,
        Use=DeviceUse.TV,
        ReleaseYear=2017
    ),
    'A1F8D55J0FWDTN': DeviceType(
        ID='A1F8D55J0FWDTN',
        CodeName='Keira/Blanche/Joyce/Tara/Rose',
        ExternalName='FTV edition TVs',
        HasScreen=True,
        Use=DeviceUse.TV,
        ReleaseYear=2018
    ),
    'A1P7E7V3FCZKU6': DeviceType(
        ID='A1P7E7V3FCZKU6',
        CodeName='Keira/Blanche/Joyce/Tara/Rose',
        ExternalName='FTV edition TVs',
        HasScreen=True,
        Use=DeviceUse.TV,
        ReleaseYear=2018
    ),
    'AP4RS91ZQ0OOI': DeviceType(
        ID='AP4RS91ZQ0OOI',
        CodeName='Keira/Blanche/Joyce/Tara/Rose',
        ExternalName='FTV edition TVs',
        HasScreen=True,
        Use=DeviceUse.TV,
        ReleaseYear=2018
    ),
    'A1GAVERFF7K6QE': DeviceType(
        ID='A1GAVERFF7K6QE',
        CodeName='Hailey Plus',
        ExternalName='2P Fire TV Devices',
        HasScreen=True,
        Use=DeviceUse.TV,
        ReleaseYear=2021,
        System='FOS7'
    ),
    'A1VGB7MHSIEYFK': DeviceType(
        ID='A1VGB7MHSIEYFK',
        CodeName='Gazelle',
        ExternalName='Fire TV Cube',
        Use=DeviceUse.TV,
        ReleaseYear=2022,
        System='FOS7'
    ),
    # endregion
}
