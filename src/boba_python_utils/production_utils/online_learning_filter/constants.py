from boba_python_utils.production_utils.common._constants.path import SupportedRegions, SupportedLocales

ONLINE_LEARNING_FILTER_PATH_PATTERNS = {
    SupportedLocales.EN_US: 's3://abml-workspaces-na/rewrite_filter/merger/unified/snapshot/4.2/{year}/{month}/{day}/{hour}/',
    SupportedRegions.NA: 's3://abml-workspaces-na/utterance-blacklist-daily-snapshots/region=na_daily/1.0.0/{year}/{month}/{day}/{hour}/locale={locale}/',
    SupportedRegions.EU: 's3://abml-workspaces-eu/utterance-blacklist-snapshots/region=eu/1.0.0/{year}/{month}/{day}/{hour}/locale={locale}/',
    SupportedRegions.FE: 's3://abml-workspaces-jp/utterance-blacklist-snapshots/region=fe/1.0.0/{year}/{month}/{day}/{hour}/locale={locale}/',
}

KEY_OLB_IS_FINALLY_BLOCKED = 'isFinallyBlocked'
KEY_OLB_PROVIDER_NAME = 'simplifiedRewriteProviderName'
KEY_OLB_VERSION = 'version'
KEY_OLB_LOCALE = 'locale'
KEY_OLB_QUERY = 'sourceUtterance'
KEY_OLB_REWRITE = 'rewriteUtterance'
KEY_OLB_REWRITE_LEGACY = 'replacedRequest'
