GW_KEY_CUSTOMER_ID = 'customerID'
GW_KEY_PERSON_ID = 'personID'
GW_KEY_TURN_TIMESTAMP = 'timestamp'
GW_KEY_TURN_LOCAL_TIMESTAMP = 'localTimestamp'
GW_KEY_DEVICE_TYPE = 'deviceType'
GW_KEY_DEVICE_ID = 'deviceId'
GW_KEY_DIALOG_ID = 'dialogId'
GW_KEY_CLIENT_PROFILE = 'clientProfile'

GW_KEY_MACAW_CUSTOMER_RESPONSE_TYPE = 'macawCustomerResponseType'
GW_KEY_MACAW_CAMPAIGN_ID = 'macawCampaignId'
GW_KEY_MACAW_ORIGINAL_UTTERANCE_ID = 'macawOriginalUtteranceId'

GW_KEY_VIDEO_DEVICE_ACTIVE_SESSION = 'deviceVideoState.activeSession'
GW_KEY_VIDEO_DEVICE_ENABLED = 'deviceVideoState.videoDeviceEnabled'

GW_KEY_IS_LLM_TRAFFIC = 'isLLMTraffic'
GW_KEY_LLM_TOKEN = 'llmToken'
GW_KEY_LLM_PROPERTIES = 'llmProperties'
GW_KEY_ALEXA_STACK_TYPE = 'stackArbitrationResult.alexaStackType'
GW_KEY_ALEXA_STACK_CONFIG = 'stackArbitrationResult.alexaStackConfig'
GW_KEY_INFO_CATEGORY = 'info_category'
GW_KEY_CHILD_DIRECT_REQUEST = 'child_directed_request'

GW_KEY_NLU_MERGE_RESULT = 'nluAusMergerResult'
GW_KEY_NLU_MERGE_DETAILED_RESULTS = 'nluAusMergerDetailedResult'
GW_KEY_NLU_MERGER_RULES = f'{GW_KEY_NLU_MERGE_DETAILED_RESULTS}.DroolsMergerDecisionAsrAus.DroolsMergerDecisionAusAsrMatchedRules'  # noqa: E501
GW_KEY_NLU_MERGER_RULES_T1 = f'{GW_KEY_NLU_MERGE_DETAILED_RESULTS}.T1_DroolsMergerDecisionAsrAus.DroolsMergerDecisionAusAsrMatchedRules'  # noqa: E501
GW_KEY_NLU_MERGER_RULES_T2 = f'{GW_KEY_NLU_MERGE_DETAILED_RESULTS}.T2_DroolsMergerDecisionAsrAus.DroolsMergerDecisionAusAsrMatchedRules'  # noqa: E501
GW_KEY_NLU_MERGER_RULES_C = f'{GW_KEY_NLU_MERGE_DETAILED_RESULTS}.C_DroolsMergerDecisionAsrAus.DroolsMergerDecisionAusAsrMatchedRules'  # noqa: E501

GW_KEY_NLU_MERGER_DETAIL_ASR_INTERPRETATION = f'{GW_KEY_NLU_MERGE_DETAILED_RESULTS}.ASRInterpretations'
GW_KEY_NLU_MERGER_DETAIL_AUS_INTERPRETATION = f'{GW_KEY_NLU_MERGE_DETAILED_RESULTS}.AUSInterpretations'

GW_KEY_TURN_CPD = 'signals.defect.value'
GW_KEY_TURN_CPD_SCORE = 'signals.defect.score'
GW_KEY_TURN_CPD_VERSION = 'signals.defect.version'
GW_KEY_TURN_DEFECT_BARGEIN = 'signals.defective_barge_in.value'
GW_KEY_TURN_DEFECT_REPHRASE = 'signals.rephrase.value'
GW_KEY_TURN_DEFECT_TERMINATION = 'signals.defective_termination.value'
GW_KEY_TURN_DEFECT_SANDPAPER = 'signals.sandpaper_friction.value'
GW_KEY_TURN_DEFECT_UNHANDLED = 'signals.unhandled_friction.value'
GW_KEY_SESSION_DEFECT = 'session_signals.total_signals.total_defect'

GW_KEY_AUS_CRITICAL = 'ausCritical'
GW_KEY_AUS_TRACE_RECORD = f'{GW_KEY_AUS_CRITICAL}.GenerateAlternativeUtterancesV2.traceRecord'
GW_KEY_LLM_SESSION_TOKEN = f'{GW_KEY_AUS_TRACE_RECORD}.simplifiedRequest.llmSessionToken'
GW_KEY_TURN_DFS_SHOULD_TRIGGER = f'{GW_KEY_AUS_TRACE_RECORD}.triggers.dfsTrigger.shouldTrigger'
GW_KEY_TURN_IS_HOLDOUT = f'{GW_KEY_NLU_MERGE_DETAILED_RESULTS}.HoldOutAusRewrite'
GW_KEY_WEBLAB = 'weblabs_information'
GW_KEY_AUS_MERGER_RESULTS = f'{GW_KEY_AUS_TRACE_RECORD}.merger'
GW_KEY_ASR_NBEST = f'{GW_KEY_AUS_TRACE_RECORD}.simplifiedRequest.originalUtterances'

GW_KEY_PIPELINES = f'{GW_KEY_AUS_TRACE_RECORD}.rawPipelineResponses'  # rawPipelineResponses
GW_KEY_DFS_SOURCE = f'{GW_KEY_PIPELINES}.Dfs.pipelineResponse.stableRewrite.alternativeUtterances.metadata.features.DFS_SOURCE'
GW_KEY_DFS_SCORE_BIN = f'{GW_KEY_PIPELINES}.Dfs.pipelineResponse.stableRewrite.alternativeUtterances.score.bin'
GW_KEY_DFS_SCORE_LATENCY = f'{GW_KEY_PIPELINES}.Dfs.pipelineExecutionDuration.latencyInMillis'
GW_KEY_DFS_SCORE = f'{GW_KEY_PIPELINES}.Dfs.pipelineResponse.stableRewrite.alternativeUtterances.score.score'
GW_KEY_DFS_ALTERNATIVE_UTTERANCES = f'{GW_KEY_PIPELINES}.Dfs.pipelineResponse.stableRewrite.alternativeUtterances'

GW_KEY_PIPELINES2 = f'{GW_KEY_AUS_TRACE_RECORD}.pipelines'  # pipelines
GW_KEY_DFS_SOURCE2 = f'{GW_KEY_PIPELINES2}.Dfs.pipelineResponse.stableRewrite.alternativeUtterances.metadata.features.DFS_SOURCE'
GW_KEY_DFS_SCORE_BIN2 = f'{GW_KEY_PIPELINES2}.Dfs.pipelineResponse.stableRewrite.alternativeUtterances.score.bin'
GW_KEY_DFS_SCORE_LATENCY2 = f'{GW_KEY_PIPELINES2}.Dfs.pipelineExecutionDuration.latencyInMillis'
GW_KEY_DFS_SCORE2 = f'{GW_KEY_PIPELINES2}.Dfs.pipelineResponse.stableRewrite.alternativeUtterances.score.score'
GW_KEY_DFS_ALTERNATIVE_UTTERANCES2 = f'{GW_KEY_PIPELINES2}.Dfs.pipelineResponse.stableRewrite.alternativeUtterances'

# region Greenwich3 Keys
GW3_KEY_TURN_TIMESTAMP = f'turn.{GW_KEY_TURN_TIMESTAMP}'
GW3_KEY_SESSION_ID = '_id'
GW3_KEY_UTTERANCE_ID = 'turn._id'
GW3_KEY_DIALOG_ID = f'turn.{GW_KEY_DIALOG_ID}'
GW3_KEY_TURN_INDEX = 'turn._index'

GW3_KEY_NLU_MERGE_RESULT = f'turn.{GW_KEY_NLU_MERGE_RESULT}'
GW3_KEY_NLU_MERGE_DETAILED_RESULTS = f'turn.{GW_KEY_NLU_MERGE_DETAILED_RESULTS}'
GW3_KEY_NLU_MERGER_RULES = f'turn.{GW_KEY_NLU_MERGER_RULES}'
GW3_KEY_NLU_MERGER_RULES_T1 = f'turn.{GW_KEY_NLU_MERGER_RULES_T1}'
GW3_KEY_NLU_MERGER_RULES_T2 = f'turn.{GW_KEY_NLU_MERGER_RULES_T2}'
GW3_KEY_NLU_MERGER_RULES_C = f'turn.{GW_KEY_NLU_MERGER_RULES_C}'

GW3_KEY_NLU_MERGER_DETAIL_ASR_INTERPRETATION = f'turn.{GW_KEY_NLU_MERGER_DETAIL_ASR_INTERPRETATION}'
GW3_KEY_NLU_MERGER_DETAIL_AUS_INTERPRETATION = f'turn.{GW_KEY_NLU_MERGER_DETAIL_AUS_INTERPRETATION}'

GW3_KEY_DOMAIN = 'turn.domain'
GW3_KEY_INTENT = 'turn.intent'
GW3_KEY_RESPONSE = 'turn.response'
GW3_KEY_TURN_DEFECT_BARGEIN = 'turn.signals.defectiveBargeIn.value'
GW3_KEY_TURN_DEFECT_REPHRASE = f'turn.{GW_KEY_TURN_DEFECT_REPHRASE}'
GW3_KEY_TURN_DEFECT_TERMINATION = 'turn.signals.defectiveTermination.value'
GW3_KEY_TURN_DEFECT_SANDPAPER = 'turn.signals.sandpaperFriction.value'
GW3_KEY_TURN_DEFECT_UNHANDLED = 'turn.signals.unhandledRequest.value'

GW3_KEY_AUS_CRITICAL = f'turn.{GW_KEY_AUS_CRITICAL}'
GW3_KEY_AUS_TRACE_RECORD = f'turn.{GW_KEY_AUS_TRACE_RECORD}'
GW3_KEY_TURN_DFS_SHOULD_TRIGGER = f'{GW3_KEY_AUS_TRACE_RECORD}.triggers.dfsTrigger.shouldTrigger'
GW3_KEY_TURN_IS_HOLDOUT = f'turn.{GW_KEY_TURN_IS_HOLDOUT}'
GW3_KEY_WEBLAB = f'turn.{GW_KEY_WEBLAB}'
GW3_KEY_AUS_MERGER_RESULTS = f'{GW3_KEY_AUS_TRACE_RECORD}.merger'
GW3_KEY_ASR_NBEST = f'turn.{GW_KEY_ASR_NBEST}'

GW3_KEY_PIPELINES = f'{GW3_KEY_AUS_TRACE_RECORD}.rawPipelineResponses'
GW3_KEY_DFS_SOURCE = f'{GW3_KEY_PIPELINES}.Dfs.pipelineResponse.stableRewrite.alternativeUtterances.metadata.features.DFS_SOURCE'
GW3_KEY_DFS_SCORE_BIN = f'{GW3_KEY_PIPELINES}.Dfs.pipelineResponse.stableRewrite.alternativeUtterances.score.bin'
GW3_KEY_DFS_SCORE_LATENCY = f'{GW3_KEY_PIPELINES}.Dfs.pipelineExecutionDuration.latencyInMillis'
GW3_KEY_DFS_SCORE = f'{GW3_KEY_PIPELINES}.Dfs.pipelineResponse.stableRewrite.alternativeUtterances.score.score'
GW3_KEY_DFS_ALTERNATIVE_UTTERANCES = f'{GW3_KEY_PIPELINES}.Dfs.pipelineResponse.stableRewrite.alternativeUtterances'
# endregion
