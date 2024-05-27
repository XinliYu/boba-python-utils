from pyspark.sql.types import (
    StructType,
    StructField,
    ArrayType,
    DoubleType,
    BooleanType,
    StringType,
    IntegerType,
    LongType,
)

ALTERNATIVE_UTTERANCES_SCHEMA = StructType(
    [
        StructField(
            'alternativeUtterances',
            ArrayType(
                StructType(
                    [
                        StructField('utterance', StringType(), True),
                        StructField(
                            'score',
                            StructType(
                                [
                                    StructField('score', DoubleType(), True),
                                    StructField('bin', StringType(), True),
                                ]
                            ),
                            True,
                        ),
                    ]
                )
            ),
            True,
        ),
        StructField(
            'metadata',
            StructType(
                [
                    StructField('mabAlpha', DoubleType(), True),
                    StructField('mabBeta', DoubleType(), True),
                    StructField('mabAlphaPrior', DoubleType(), True),
                    StructField('mabBetaPrior', DoubleType(), True),
                    StructField('providerName', StringType(), True),
                    StructField('providerVersion', StringType(), True),
                ]
            ),
            True,
        ),
    ]
)

PIPELINE_EXECUTION_DURATION_SCHEMA = StructType(
    [
        StructField('startTime', StringType(), True),
        StructField('endTime', StringType(), True),
        StructField('latencyInMillis', IntegerType(), True),
    ]
)

PIPELINE_RESPONSE_SCHEMA = StructType(
    [
        StructField(
            'experimentalRewrites',
            ArrayType(
                StructType(
                    [
                        StructField('experimentName', StringType(), True),
                        StructField(
                            'candidates',
                            StructType(
                                [
                                    StructField('C', ALTERNATIVE_UTTERANCES_SCHEMA, True),
                                    StructField('T1', ALTERNATIVE_UTTERANCES_SCHEMA, True),
                                ]
                            ),
                            True,
                        ),
                    ]
                )
            ),
            True,
        ),
        StructField(
            'stableRewrite',
            StructType(
                [
                    StructField(
                        'alternativeUtterances',
                        ArrayType(
                            StructType(
                                [
                                    StructField('utterance', StringType(), True),
                                    StructField(
                                        'score',
                                        StructType(
                                            [
                                                StructField('score', DoubleType(), True),
                                                StructField('bin', StringType(), True),
                                            ]
                                        ),
                                        True,
                                    ),
                                    StructField(
                                        'metadata',
                                        StructType(
                                            [
                                                StructField('providerName', StringType(), True),
                                                StructField('providerVersion', StringType(), True),
                                                StructField('rewriteType', StringType(), True),
                                                StructField(
                                                    'features',
                                                    StructType(
                                                        [
                                                            StructField(
                                                                'DFS_SOURCE', StringType(), True
                                                            ),
                                                        ]
                                                    ),
                                                    True,
                                                ),
                                            ]
                                        ),
                                        True,
                                    ),
                                ]
                            )
                        ),
                        True,
                    ),
                ]
            ),
            True,
        ),
    ]
)

REWRITE_PIPELINE_SCHEMA = StructType(
    [
        StructField('weblabAssignment', StringType(), True),
        StructField('weblabName', StringType(), True),
        StructField('pipelineResponse', PIPELINE_RESPONSE_SCHEMA, True),
        StructField('pipelineExecutionDuration', PIPELINE_EXECUTION_DURATION_SCHEMA, True),
        StructField('processedResponse', ALTERNATIVE_UTTERANCES_SCHEMA, False),
    ]
)

PIPELINE_SCHEMA = StructType(
    [
        StructField('weblabName', StringType(), True),
        StructField('weblabAssignment', StringType(), True),
        StructField('shouldPreventRewrite', BooleanType(), True),
        StructField('Dfs', REWRITE_PIPELINE_SCHEMA, True),
    ]
)

TRACE_RECORD_SCHEMA = StructType(
    [
        StructField(
            "simplifiedRequest",
            StructType(
                [
                    StructField(
                        "originalUtterances",
                        ArrayType(
                            StructType(
                                [
                                    StructField('utterance', StringType(), True),
                                    StructField('asrConfidence', IntegerType(), True),
                                    StructField('asrDirectness', IntegerType(), True),
                                    StructField(
                                        'wakeWordToken',
                                        StructType(
                                            [
                                                StructField('word', StringType(), True),
                                                StructField('confidenceScore', IntegerType(), True),
                                            ]
                                        ),
                                        True,
                                    ),
                                ]
                            )
                        ),
                        True,
                    ),
                ]
            ),
            True,
        ),
        StructField(
            'triggers',
            StructType(
                [
                    StructField(
                        'dfsTrigger',
                        StructType(
                            [
                                StructField('shouldTrigger', StringType(), True),
                            ]
                        ),
                        True,
                    ),
                ]
            ),
            True,
        ),
        StructField('rawPipelineResponses', PIPELINE_SCHEMA, True),
        StructField(
            'merger',
            ArrayType(
                StructType(
                    [
                        StructField(
                            'alternativeUtterance',
                            StructType(
                                [
                                    StructField(
                                        'metadata',
                                        StructType(
                                            [
                                                StructField('providerName', StringType(), True),
                                                StructField('providerVersion', StringType(), True),
                                            ]
                                        ),
                                        True,
                                    ),
                                ]
                            ),
                            True,
                        ),
                    ]
                )
            ),
            True,
        ),
    ]
)

AUS_CRITICAL_SCHEMA = StructType(
    [
        StructField(
            'GenerateAlternativeUtterancesV2',
            StructType(
                [
                    StructField('traceRecord', TRACE_RECORD_SCHEMA, True),
                    StructField(
                        'response',
                        ArrayType(
                            StructType(
                                [
                                    StructField(
                                        'metadata',
                                        StructType(
                                            [
                                                StructField(
                                                    'generation',
                                                    StructType(
                                                        [
                                                            StructField(
                                                                'dimensions',
                                                                StructType(
                                                                    [
                                                                        StructField(
                                                                            'searchBased',
                                                                            StringType(),
                                                                            True,
                                                                        )
                                                                    ]
                                                                ),
                                                                True,
                                                            ),
                                                        ]
                                                    ),
                                                    True,
                                                ),
                                            ]
                                        ),
                                        True,
                                    ),
                                ]
                            )
                        ),
                        True,
                    ),
                ]
            ),
            True,
        ),
    ]
)

NLU_MERGER_DETAIL_SCHEMA = StructType(
    [
        StructField('HoldOutAusRewrite', StringType(), True),
        StructField('AUSInterpretations', StringType(), True),
        StructField('ASRInterpretations', StringType(), True),
        StructField('Reason', StringType(), True),
        StructField('MergerDecision', StringType(), True),
        StructField('T1_decision', StringType(), True),
        StructField('T2_decision', StringType(), True),
        StructField(
            'DroolsMergerDecisionAsrAus',
            StructType(
                [
                    StructField(
                        'DroolsMergerDecisionAusAsrMatchedRules',
                        ArrayType(
                            StructType(
                                [
                                    StructField('ruleName', StringType(), True),
                                    StructField('ruleGroup', StringType(), True),
                                ]
                            )
                        ),
                        True,
                    ),
                ]
            ),
            True,
        ),
        StructField(
            'C_DroolsMergerDecisionAsrAus',
            StructType([
                StructField(
                    'DroolsMergerDecisionAusAsrMatchedRules',
                    ArrayType(
                        StructType(
                            [
                                StructField('ruleName', StringType(), True),
                                StructField('ruleGroup', StringType(), True)
                            ]
                        )),
                    True
                )
            ]), True),
        StructField(
            'T1_DroolsMergerDecisionAsrAus',
            StructType([
                StructField(
                    'DroolsMergerDecisionAusAsrMatchedRules',
                    ArrayType(
                        StructType(
                            [
                                StructField('ruleName', StringType(), True),
                                StructField('ruleGroup', StringType(), True)
                            ]
                        )),
                    True
                )
            ]), True),
        StructField(
            'T2_DroolsMergerDecisionAsrAus',
            StructType([
                StructField(
                    'DroolsMergerDecisionAusAsrMatchedRules',
                    ArrayType(
                        StructType(
                            [
                                StructField('ruleName', StringType(), True),
                                StructField('ruleGroup', StringType(), True)
                            ]
                        )),
                    True
                )
            ]), True),
    ]
)

GREENWICH_SIGNALS_STRUCT = StructField(
    "signals",
    StructType(
        [
            StructField(
                "greenwichDefect", StructType([StructField("value", IntegerType(), True)]), True
            ),
            StructField(
                "perceivedDefect",
                StructType(
                    [
                        StructField("value", IntegerType(), True),
                    ]
                ),
                True,
            ),
            StructField(
                "defectiveBargeIn", StructType([StructField("value", IntegerType(), True)]), True
            ),
            StructField("rephrase", StructType([StructField("value", IntegerType(), True)]), True),
            StructField(
                "defectiveTermination",
                StructType([StructField("value", IntegerType(), True)]),
                True,
            ),
            StructField(
                "sandpaperFriction", StructType([StructField("value", IntegerType(), True)]), True
            ),
            StructField(
                "unhandledRequest", StructType([StructField("value", IntegerType(), True)]), True
            ),
        ]
    ),
)

SCHEMA_GREENWICH3 = StructType(
    [
        StructField("_id", StringType(), True),
        StructField("customerId", StringType(), True),
        StructField('locale', StringType(), False),
        StructField('userGroup', StringType(), False),
        StructField(
            'signals',
            StructType(
                [
                    StructField('totalTurns', IntegerType(), True),
                    StructField(
                        'greenwichDefects',
                        StructType([StructField('totalDefects', IntegerType(), True)]),
                        True,
                    ),
                    StructField(
                        'perceivedDefects',
                        StructType([StructField('totalDefects', IntegerType(), True)]),
                        True,
                    ),
                ]
            ),
            True,
        ),
        StructField(
            "turns",
            ArrayType(
                StructType(
                    [
                        StructField("_id", StringType(), True),
                        StructField("_index", IntegerType(), True),
                        StructField('dialogId', StringType(), True),
                        StructField("domain", StringType(), True),
                        StructField("intent", StringType(), True),
                        StructField(
                            "slots",
                            ArrayType(
                                StructType(
                                    [
                                        StructField("slotName", StringType(), True),
                                        StructField("slotValue", StringType(), True),
                                        StructField("tokens", ArrayType(StringType()), True),
                                    ]
                                )
                            ),
                            True,
                        ),
                        StructField("request", ArrayType(StringType()), True),
                        StructField("replacedRequest", ArrayType(StringType()), True),
                        StructField("response", StringType(), True),
                        StructField("tokenLabelText", StringType(), True),
                        StructField("timestamp", StringType(), True),
                        StructField("nluAusMergerResult", StringType(), True),
                        StructField("nluAusMergerDetailedResult", NLU_MERGER_DETAIL_SCHEMA, True),
                        StructField("ausCritical", AUS_CRITICAL_SCHEMA, True),
                        GREENWICH_SIGNALS_STRUCT,
                        StructField("weblabs_information", StringType(), True),
                        StructField(
                            "flare_logger",
                            StructType(
                                [
                                    StructField("response_length", IntegerType(), True),
                                    StructField(
                                        "filtered_candidates_agg_count",
                                        StructType(
                                            [
                                                StructField(
                                                    "ScoreValidationFilter", IntegerType(), True
                                                ),
                                                StructField(
                                                    "LowConfidenceFilter", IntegerType(), True
                                                ),
                                                StructField(
                                                    "IdenticalRewriteFilter", IntegerType(), True
                                                ),
                                                StructField(
                                                    "SingleResultFilter", IntegerType(), True
                                                ),
                                                StructField(
                                                    "GlobalRewriteFilter", IntegerType(), True
                                                ),
                                                StructField(
                                                    "PersonalizedRewriteFilter", IntegerType(), True
                                                ),
                                                StructField("ProfanityFilter", IntegerType(), True),
                                                StructField(
                                                    "RealTimeDefectPreventionFilter",
                                                    IntegerType(),
                                                    True,
                                                ),
                                                StructField(
                                                    "MultiArmedBanditRanker", IntegerType(), True
                                                ),
                                                StructField(
                                                    "PercentageLimiter", IntegerType(), True
                                                ),
                                            ]
                                        ),
                                        True,
                                    ),
                                ]
                            ),
                            True,
                        ),
                    ]
                )
            ),
        ),
    ]
)

GREENWICH_SCHEMA_LIGHT = StructType(
    [
        StructField("_id", StringType(), True),
        StructField("customerId", StringType(), True),
        StructField('locale', StringType(), False),
        StructField('userGroup', StringType(), False),
        StructField(
            "turns",
            ArrayType(
                StructType(
                    [
                        StructField("_id", StringType(), True),
                        StructField("_index", IntegerType(), True),
                        StructField('dialogId', StringType(), True),
                        StructField("domain", StringType(), True),
                        StructField("intent", StringType(), True),
                        StructField("request", ArrayType(StringType()), True),
                        StructField("replacedRequest", ArrayType(StringType()), True),
                        StructField("response", StringType(), True),
                        StructField("tokenLabelText", StringType(), True),
                        StructField("timestamp", StringType(), True),
                        StructField("nluAusMergerResult", StringType(), True),
                        StructField("nluAusMergerDetailedResult", NLU_MERGER_DETAIL_SCHEMA, True),
                    ]
                )
            ),
        ),
    ]
)

SCHEMA_GREENWICH_NEXTGEN = StructType(
    [
        StructField("timestamp", StringType(), True),
        StructField("customerId", StringType(), True),
        StructField("deviceType", StringType(), True),
        StructField("deviceId", StringType(), True),
        StructField("locale", StringType(), True),
        StructField("userGroup", StringType(), True),
        StructField("_id", StringType(), True),
        StructField(
            "session_signals",
            ArrayType(
                StructType(
                    [
                        StructField("cpd_version", StringType(), True),
                        StructField(
                            "total_signals",
                            StructType([StructField("total_defect", DoubleType(), True)]),
                        ),
                    ]
                )
            ),
        ),
        StructField(
            "turns",
            ArrayType(
                StructType(
                    [
                        StructField("_index", LongType(), True),
                        StructField("_id", StringType(), True),
                        StructField("timestamp", StringType(), True),
                        StructField("domain", StringType(), True),
                        StructField("intent", StringType(), True),
                        StructField("dialogId", StringType(), True),
                        StructField("dialogStatusCode", StringType(), True),
                        StructField("request", ArrayType(StringType()), True),
                        StructField("response", StringType(), True),
                        StructField("tokenLabelText", StringType(), True),
                        StructField("replacedRequest", ArrayType(StringType()), True),
                        StructField("nluAusMergerResult", StringType(), True),
                        StructField(
                            "signals",
                            StructType(
                                [
                                    StructField(
                                        "rephrase",
                                        StructType(
                                            [
                                                StructField(
                                                    "versions",
                                                    ArrayType(
                                                        StructType(
                                                            [
                                                                StructField(
                                                                    "version", StringType(), True
                                                                ),
                                                                StructField(
                                                                    "score", DoubleType(), True
                                                                ),
                                                                StructField(
                                                                    "value", DoubleType(), True
                                                                ),
                                                                StructField(
                                                                    "distance", DoubleType(), True
                                                                ),
                                                                StructField(
                                                                    "score_bin", StringType(), True
                                                                ),
                                                                StructField(
                                                                    "delay", DoubleType(), True
                                                                ),
                                                            ]
                                                        )
                                                    ),
                                                )
                                            ]
                                        ),
                                    ),
                                    StructField(
                                        "barge_in",
                                        StructType(
                                            [
                                                StructField(
                                                    "versions",
                                                    ArrayType(
                                                        StructType(
                                                            [
                                                                StructField(
                                                                    "version", StringType(), True
                                                                ),
                                                                StructField(
                                                                    "score", DoubleType(), True
                                                                ),
                                                                StructField(
                                                                    "value", DoubleType(), True
                                                                ),
                                                            ]
                                                        )
                                                    ),
                                                )
                                            ]
                                        ),
                                    ),
                                    StructField(
                                        "termination",
                                        StructType(
                                            [
                                                StructField(
                                                    "versions",
                                                    ArrayType(
                                                        StructType(
                                                            [
                                                                StructField(
                                                                    "version", StringType(), True
                                                                ),
                                                                StructField(
                                                                    "score", DoubleType(), True
                                                                ),
                                                                StructField(
                                                                    "value", DoubleType(), True
                                                                ),
                                                            ]
                                                        )
                                                    ),
                                                )
                                            ]
                                        ),
                                    ),
                                    StructField(
                                        "sandpaper_friction",
                                        StructType(
                                            [
                                                StructField(
                                                    "versions",
                                                    ArrayType(
                                                        StructType(
                                                            [
                                                                StructField(
                                                                    "version", StringType(), True
                                                                ),
                                                                StructField(
                                                                    "score", DoubleType(), True
                                                                ),
                                                                StructField(
                                                                    "value", DoubleType(), True
                                                                ),
                                                                StructField(
                                                                    "userActionRequired",
                                                                    LongType(),
                                                                    True,
                                                                ),
                                                                StructField(
                                                                    "unsupportedUseCase",
                                                                    LongType(),
                                                                    True,
                                                                ),
                                                                StructField(
                                                                    "coverageGap", LongType(), True
                                                                ),
                                                                StructField(
                                                                    "errorOrRetry", LongType(), True
                                                                ),
                                                                StructField(
                                                                    "noReply", StringType(), True
                                                                ),
                                                            ]
                                                        )
                                                    ),
                                                )
                                            ]
                                        ),
                                    ),
                                    StructField(
                                        "query_response_quality",
                                        StructType(
                                            [
                                                StructField(
                                                    "versions",
                                                    ArrayType(
                                                        StructType(
                                                            [
                                                                StructField(
                                                                    "version", StringType(), True
                                                                ),
                                                                StructField(
                                                                    "score", DoubleType(), True
                                                                ),
                                                                StructField(
                                                                    "value", DoubleType(), True
                                                                ),
                                                            ]
                                                        )
                                                    ),
                                                )
                                            ]
                                        ),
                                    ),
                                    StructField(
                                        "bert",
                                        StructType(
                                            [
                                                StructField(
                                                    "versions",
                                                    ArrayType(
                                                        StructType(
                                                            [
                                                                StructField(
                                                                    "version", StringType(), True
                                                                ),
                                                                StructField(
                                                                    "score", DoubleType(), True
                                                                ),
                                                                StructField(
                                                                    "value", DoubleType(), True
                                                                ),
                                                            ]
                                                        )
                                                    ),
                                                )
                                            ]
                                        ),
                                    ),
                                    StructField(
                                        "defect",
                                        StructType(
                                            [
                                                StructField(
                                                    "versions",
                                                    ArrayType(
                                                        StructType(
                                                            [
                                                                StructField(
                                                                    "version", StringType(), True
                                                                ),
                                                                StructField(
                                                                    "score", DoubleType(), True
                                                                ),
                                                                StructField(
                                                                    "value", DoubleType(), True
                                                                ),
                                                            ]
                                                        )
                                                    ),
                                                )
                                            ]
                                        ),
                                    ),
                                    StructField(
                                        "defective_barge_in",
                                        StructType(
                                            [
                                                StructField(
                                                    "versions",
                                                    ArrayType(
                                                        StructType(
                                                            [
                                                                StructField(
                                                                    "version", StringType(), True
                                                                ),
                                                                StructField(
                                                                    "score", DoubleType(), True
                                                                ),
                                                                StructField(
                                                                    "value", DoubleType(), True
                                                                ),
                                                            ]
                                                        )
                                                    ),
                                                )
                                            ]
                                        ),
                                    ),
                                    StructField(
                                        "defective_termination",
                                        StructType(
                                            [
                                                StructField(
                                                    "versions",
                                                    ArrayType(
                                                        StructType(
                                                            [
                                                                StructField(
                                                                    "version", StringType(), True
                                                                ),
                                                                StructField(
                                                                    "score", DoubleType(), True
                                                                ),
                                                                StructField(
                                                                    "value", DoubleType(), True
                                                                ),
                                                            ]
                                                        )
                                                    ),
                                                )
                                            ]
                                        ),
                                    ),
                                    StructField(
                                        "defective_rephrase",
                                        StructType(
                                            [
                                                StructField(
                                                    "versions",
                                                    ArrayType(
                                                        StructType(
                                                            [
                                                                StructField(
                                                                    "version", StringType(), True
                                                                ),
                                                                StructField(
                                                                    "score", DoubleType(), True
                                                                ),
                                                                StructField(
                                                                    "value", DoubleType(), True
                                                                ),
                                                            ]
                                                        )
                                                    ),
                                                )
                                            ]
                                        ),
                                    ),
                                    StructField(
                                        "unhandled_friction",
                                        StructType(
                                            [
                                                StructField(
                                                    "versions",
                                                    ArrayType(
                                                        StructType(
                                                            [
                                                                StructField(
                                                                    "version", StringType(), True
                                                                ),
                                                                StructField(
                                                                    "score", DoubleType(), True
                                                                ),
                                                                StructField(
                                                                    "value", DoubleType(), True
                                                                ),
                                                            ]
                                                        )
                                                    ),
                                                )
                                            ]
                                        ),
                                    ),
                                ]
                            ),
                        ),
                    ]
                )
            ),
        ),
    ]
)
