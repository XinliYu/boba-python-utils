from typing import Union, Iterable, Tuple, Optional

from attr import attrs, attrib
from pyspark.sql.functions import col

from boba_python_utils.general_utils.common_objects.filter_config import FilterCondConfig
from boba_python_utils.common_utils.typing_helper import make_set
from boba_python_utils.spark_utils.spark_functions.common import and_, or_
from boba_python_utils.production_utils.pdfs import constants as c


@attrs(slots=True)
class EntityDefectGuardrailConfig(FilterCondConfig):
    entity_types = attrib(type=Union[str, Iterable[str]])
    impression_defect_guardrail_thresholds = attrib(type=Iterable[Tuple[int, float]])
    enable_guardrail_min_impression = attrib(type=Optional[int])

    def get_filter_cond(self, **kwargs):
        entity_type_colname = kwargs.get('entity_type_colname', c.KEY_ENTITY_TYPE)
        defect_colname = kwargs.get('defect_colname', c.KEY_CUSTOMER_AVG_DEFECT)
        count_colname = kwargs.get('count_colname', c.KEY_CUSTOMER_COUNT)

        return and_(
            col(entity_type_colname).isin(make_set(self.entity_types)),
            or_(
                (
                    (
                            (col(count_colname) >= count_th) &
                            (col(defect_colname) <= defect_th)
                    ) if count_th is not None
                    else (col(defect_colname) <= defect_th)
                )
                for count_th, defect_th in self.impression_defect_guardrail_thresholds
            ),
            (
                    col(count_colname) >= self.enable_guardrail_min_impression
            ) if self.enable_guardrail_min_impression is not None else None
        )


CUSTOMER_ENTITY_DEFECT_GUARDRAIL_CONFIGS = (
    EntityDefectGuardrailConfig(
        entity_types={
            'AppName',
            'ArtistName',
            'SongName',
            'PlaylistName',
            'GenreName',
            'CallSign',
            'StationName',
            'ServiceName',
            'ProgramName',
            'ContentSourceName',
            'BookName',
            'DeviceBrand',
            'DeviceLocation',
            'ChannelName',
            'VideoName',
            'ContactName',
            'ItemName',
            'Language',
            'TopicName',
            'DishName',
            'ShoppingServiceName',
            'PlaceName',
            'WeatherDuration',
            'RoutineName',
            'LocationName',
            'Country'
        },
        impression_defect_guardrail_thresholds=[
            (None, 0.08),
            (15, 0.12)
        ],
        enable_guardrail_min_impression=2
    ),
    EntityDefectGuardrailConfig(
        entity_types='NotificationLabel',
        impression_defect_guardrail_thresholds=[
            (None, 0.03),
            (20, 0.08)
        ],
        enable_guardrail_min_impression=3
    ),
    EntityDefectGuardrailConfig(
        entity_types='DeviceName',
        impression_defect_guardrail_thresholds=[
            (20, 0.03),
            (50, 0.19)
        ],
        enable_guardrail_min_impression=None
    )
)
