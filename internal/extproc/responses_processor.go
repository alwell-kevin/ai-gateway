package extproc

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"strconv"

	corev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extprocv3http "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	extprocv3 "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/envoyproxy/ai-gateway/filterapi"
	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/extproc/backendauth"
	"github.com/envoyproxy/ai-gateway/internal/extproc/headermutator"
	"github.com/envoyproxy/ai-gateway/internal/extproc/translator"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
	"github.com/envoyproxy/ai-gateway/internal/metrics"
	tracing "github.com/envoyproxy/ai-gateway/internal/tracing/api"
)

// ResponsesProcessorFactory returns a factory method to instantiate the responses processor.
func ResponsesProcessorFactory(ccm metrics.ChatCompletionMetrics) ProcessorFactory {
	return func(config *processorConfig, requestHeaders map[string]string, logger *slog.Logger, _ tracing.Tracing, isUpstreamFilter bool) (Processor, error) {
		logger = logger.With("processor", "responses", "isUpstreamFilter", fmt.Sprintf("%v", isUpstreamFilter))
		if !isUpstreamFilter {
			return &responsesProcessorRouterFilter{
				config:         config,
				requestHeaders: requestHeaders,
				logger:         logger,
			}, nil
		}
		return &responsesProcessorUpstreamFilter{
			config:         config,
			requestHeaders: requestHeaders,
			logger:         logger,
			metrics:        ccm,
		}, nil
	}
}

type responsesProcessorRouterFilter struct {
	passThroughProcessor
	upstreamFilter         Processor
	logger                 *slog.Logger
	config                 *processorConfig
	requestHeaders         map[string]string
	originalRequestBody    *openai.ResponsesRequest
	originalRequestBodyRaw []byte
	upstreamFilterCount    int
}

func (r *responsesProcessorRouterFilter) ProcessRequestHeaders(_ context.Context, _ *corev3.HeaderMap) (*extprocv3.ProcessingResponse, error) {
	return &extprocv3.ProcessingResponse{Response: &extprocv3.ProcessingResponse_RequestHeaders{}}, nil
}

func (r *responsesProcessorRouterFilter) ProcessRequestBody(_ context.Context, rawBody *extprocv3.HttpBody) (*extprocv3.ProcessingResponse, error) {
	model, body, err := parseOpenAIResponsesBody(rawBody)
	if err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}

	r.requestHeaders[r.config.modelNameHeaderKey] = model

	additionalHeaders := []*corev3.HeaderValueOption{
		{
			Header: &corev3.HeaderValue{Key: r.config.modelNameHeaderKey, RawValue: []byte(model)},
		},
		{
			Header: &corev3.HeaderValue{Key: originalPathHeader, RawValue: []byte(r.requestHeaders[":path"])},
		},
	}

	r.originalRequestBody = body
	r.originalRequestBodyRaw = rawBody.Body

	return &extprocv3.ProcessingResponse{
		Response: &extprocv3.ProcessingResponse_RequestBody{
			RequestBody: &extprocv3.BodyResponse{
				Response: &extprocv3.CommonResponse{
					HeaderMutation:  &extprocv3.HeaderMutation{SetHeaders: additionalHeaders},
					ClearRouteCache: true,
				},
			},
		},
	}, nil
}

func (r *responsesProcessorRouterFilter) ProcessResponseHeaders(ctx context.Context, headers *corev3.HeaderMap) (*extprocv3.ProcessingResponse, error) {
	if r.upstreamFilter != nil {
		return r.upstreamFilter.ProcessResponseHeaders(ctx, headers)
	}
	return r.passThroughProcessor.ProcessResponseHeaders(ctx, headers)
}

func (r *responsesProcessorRouterFilter) ProcessResponseBody(ctx context.Context, body *extprocv3.HttpBody) (*extprocv3.ProcessingResponse, error) {
	if r.upstreamFilter != nil {
		return r.upstreamFilter.ProcessResponseBody(ctx, body)
	}
	return r.passThroughProcessor.ProcessResponseBody(ctx, body)
}

func (r *responsesProcessorRouterFilter) SetBackend(_ context.Context, _ *filterapi.Backend, _ backendauth.Handler, _ Processor) error {
	return nil
}

type responsesProcessorUpstreamFilter struct {
	logger                 *slog.Logger
	config                 *processorConfig
	requestHeaders         map[string]string
	responseHeaders        map[string]string
	responseEncoding       string
	modelNameOverride      string
	backendName            string
	handler                backendauth.Handler
	headerMutator          *headermutator.HeaderMutator
	originalRequestBodyRaw []byte
	originalRequestBody    *openai.ResponsesRequest
	translator             translator.OpenAIResponsesTranslator
	onRetry                bool
	costs                  translator.LLMTokenUsage
	metrics                metrics.ChatCompletionMetrics
	stream                 bool
}

func (r *responsesProcessorUpstreamFilter) selectTranslator(out filterapi.VersionedAPISchema) error {
	switch out.Name {
	case filterapi.APISchemaOpenAI:
		r.translator = translator.NewResponsesOpenAIToOpenAITranslator(out.Version, r.modelNameOverride)
	default:
		return fmt.Errorf("unsupported API schema: backend=%s", out)
	}
	return nil
}

func (r *responsesProcessorUpstreamFilter) ProcessRequestHeaders(ctx context.Context, _ *corev3.HeaderMap) (resp *extprocv3.ProcessingResponse, err error) {
	defer func() {
		if err != nil {
			r.metrics.RecordRequestCompletion(ctx, false, r.requestHeaders)
		}
	}()

	r.metrics.StartRequest(r.requestHeaders)
	r.metrics.SetModel(r.requestHeaders[r.config.modelNameHeaderKey])

	headerMutation, bodyMutation, err := r.translator.RequestBody(r.originalRequestBodyRaw, r.originalRequestBody, r.onRetry)
	if err != nil {
		return nil, fmt.Errorf("failed to transform request: %w", err)
	}

	if headerMutation == nil {
		headerMutation = &extprocv3.HeaderMutation{}
	}

	if h := r.headerMutator; h != nil {
		if hm := h.Mutate(r.requestHeaders, r.onRetry); hm != nil {
			headerMutation.RemoveHeaders = append(headerMutation.RemoveHeaders, hm.RemoveHeaders...)
			headerMutation.SetHeaders = append(headerMutation.SetHeaders, hm.SetHeaders...)
		}
	}

	for _, h := range headerMutation.SetHeaders {
		r.requestHeaders[h.Header.Key] = string(h.Header.RawValue)
	}

	if h := r.handler; h != nil {
		if err = h.Do(ctx, r.requestHeaders, headerMutation, bodyMutation); err != nil {
			return nil, fmt.Errorf("failed to do auth request: %w", err)
		}
	}

	var dm *structpb.Struct
	if bm := bodyMutation.GetBody(); bm != nil {
		dm = buildContentLengthDynamicMetadataOnRequest(r.config, len(bm))
	}

	return &extprocv3.ProcessingResponse{
		Response: &extprocv3.ProcessingResponse_RequestHeaders{
			RequestHeaders: &extprocv3.HeadersResponse{
				Response: &extprocv3.CommonResponse{
					HeaderMutation: headerMutation,
					BodyMutation:   bodyMutation,
					Status:         extprocv3.CommonResponse_CONTINUE_AND_REPLACE,
				},
			},
		},
		DynamicMetadata: dm,
	}, nil
}

func (r *responsesProcessorUpstreamFilter) ProcessRequestBody(context.Context, *extprocv3.HttpBody) (*extprocv3.ProcessingResponse, error) {
	panic("BUG: ProcessRequestBody should not be called in the upstream filter")
}

func (r *responsesProcessorUpstreamFilter) ProcessResponseHeaders(ctx context.Context, headers *corev3.HeaderMap) (res *extprocv3.ProcessingResponse, err error) {
	defer func() {
		if err != nil {
			r.metrics.RecordRequestCompletion(ctx, false, r.requestHeaders)
		}
	}()

	r.responseHeaders = headersToMap(headers)
	if enc := r.responseHeaders["content-encoding"]; enc != "" {
		r.responseEncoding = enc
	}

	headerMutation, err := r.translator.ResponseHeaders(r.responseHeaders)
	if err != nil {
		return nil, fmt.Errorf("failed to transform response headers: %w", err)
	}

	var mode *extprocv3http.ProcessingMode
	if r.stream && r.responseHeaders[":status"] == "200" {
		mode = &extprocv3http.ProcessingMode{ResponseBodyMode: extprocv3http.ProcessingMode_STREAMED}
	}

	return &extprocv3.ProcessingResponse{Response: &extprocv3.ProcessingResponse_ResponseHeaders{
		ResponseHeaders: &extprocv3.HeadersResponse{
			Response: &extprocv3.CommonResponse{HeaderMutation: headerMutation},
		},
	}, ModeOverride: mode}, nil
}

func (r *responsesProcessorUpstreamFilter) ProcessResponseBody(ctx context.Context, body *extprocv3.HttpBody) (resp *extprocv3.ProcessingResponse, err error) {
	defer func() {
		r.metrics.RecordRequestCompletion(ctx, err == nil, r.requestHeaders)
	}()

	var reader io.Reader
	var isGzip bool
	switch r.responseEncoding {
	case "gzip":
		reader, err = gzip.NewReader(bytes.NewReader(body.Body))
		if err != nil {
			return nil, fmt.Errorf("failed to decode gzip: %w", err)
		}
		isGzip = true
	default:
		reader = bytes.NewReader(body.Body)
	}

	if code, _ := strconv.Atoi(r.responseHeaders[":status"]); !isGoodStatusCode(code) {
		headerMutation, bodyMutation, err := r.translator.ResponseError(r.responseHeaders, reader)
		if err != nil {
			return nil, fmt.Errorf("failed to transform response error: %w", err)
		}
		return &extprocv3.ProcessingResponse{
			Response: &extprocv3.ProcessingResponse_ResponseBody{
				ResponseBody: &extprocv3.BodyResponse{
					Response: &extprocv3.CommonResponse{
						HeaderMutation: headerMutation,
						BodyMutation:   bodyMutation,
					},
				},
			},
		}, nil
	}

	headerMutation, bodyMutation, tokenUsage, latencyTokens, err := r.translator.ResponseBody(
		r.responseHeaders, reader, body.EndOfStream,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to transform response: %w", err)
	}
	if bodyMutation != nil && isGzip {
		if headerMutation == nil {
			headerMutation = &extprocv3.HeaderMutation{}
		}
		headerMutation.RemoveHeaders = append(headerMutation.RemoveHeaders, "content-encoding")
	}

	resp = &extprocv3.ProcessingResponse{
		Response: &extprocv3.ProcessingResponse_ResponseBody{
			ResponseBody: &extprocv3.BodyResponse{
				Response: &extprocv3.CommonResponse{
					HeaderMutation: headerMutation,
					BodyMutation:   bodyMutation,
				},
			},
		},
	}

	r.costs.InputTokens += tokenUsage.InputTokens
	r.costs.OutputTokens += tokenUsage.OutputTokens
	r.costs.TotalTokens += tokenUsage.TotalTokens

	r.metrics.RecordTokenUsage(ctx, tokenUsage.InputTokens, tokenUsage.OutputTokens, tokenUsage.TotalTokens, r.requestHeaders)
	if r.stream && latencyTokens > 0 {
		r.metrics.RecordTokenLatency(ctx, latencyTokens, r.requestHeaders)
	}

	if body.EndOfStream && len(r.config.requestCosts) > 0 {
		metadata, err := buildDynamicMetadata(r.config, &r.costs, r.requestHeaders, r.modelNameOverride, r.backendName)
		if err != nil {
			return nil, fmt.Errorf("failed to build dynamic metadata: %w", err)
		}
		if r.stream {
			r.mergeWithTokenLatencyMetadata(metadata)
		}
		resp.DynamicMetadata = metadata
	}

	return resp, nil
}

func (r *responsesProcessorUpstreamFilter) SetBackend(ctx context.Context, backend *filterapi.Backend, backendHandler backendauth.Handler, routeProcessor Processor) (err error) {
	defer func() {
		r.metrics.RecordRequestCompletion(ctx, err == nil, r.requestHeaders)
	}()

	pickedEndpoint, isEndpointPicker := r.requestHeaders[internalapi.EndpointPickerHeaderKey]

	rp, ok := routeProcessor.(*responsesProcessorRouterFilter)
	if !ok {
		panic("BUG: expected routeProcessor to be of type *responsesProcessorRouterFilter")
	}
	rp.upstreamFilterCount++

	r.metrics.SetBackend(backend)
	r.modelNameOverride = backend.ModelNameOverride
	r.backendName = backend.Name
	if err = r.selectTranslator(backend.Schema); err != nil {
		return fmt.Errorf("failed to select translator: %w", err)
	}

	r.handler = backendHandler
	r.headerMutator = headermutator.NewHeaderMutator(backend.HeaderMutation, rp.requestHeaders)
	if r.modelNameOverride != "" {
		r.requestHeaders[r.config.modelNameHeaderKey] = r.modelNameOverride
	}

	r.originalRequestBody = rp.originalRequestBody
	r.originalRequestBodyRaw = rp.originalRequestBodyRaw
	r.onRetry = rp.upstreamFilterCount > 1
	if rp.originalRequestBody != nil {
		r.stream = rp.originalRequestBody.Stream
	}

	if isEndpointPicker && r.logger.Enabled(ctx, slog.LevelDebug) {
		r.logger.Debug("selected backend", slog.String("picked_endpoint", pickedEndpoint), slog.String("backendName", backend.Name), slog.String("modelNameOverride", r.modelNameOverride))
	}

	rp.upstreamFilter = r
	return nil
}

func (r *responsesProcessorUpstreamFilter) mergeWithTokenLatencyMetadata(metadata *structpb.Struct) {
	timeToFirstTokenMs := r.metrics.GetTimeToFirstTokenMs()
	interTokenLatencyMs := r.metrics.GetInterTokenLatencyMs()
	ns := r.config.metadataNamespace
	innerVal := metadata.Fields[ns].GetStructValue()
	if innerVal == nil {
		innerVal = &structpb.Struct{Fields: map[string]*structpb.Value{}}
		metadata.Fields[ns] = structpb.NewStructValue(innerVal)
	}
	innerVal.Fields["token_latency_ttft"] = &structpb.Value{Kind: &structpb.Value_NumberValue{NumberValue: timeToFirstTokenMs}}
	innerVal.Fields["token_latency_itl"] = &structpb.Value{Kind: &structpb.Value_NumberValue{NumberValue: interTokenLatencyMs}}
}

func parseOpenAIResponsesBody(body *extprocv3.HttpBody) (string, *openai.ResponsesRequest, error) {
	var request openai.ResponsesRequest
	if err := json.Unmarshal(body.Body, &request); err != nil {
		return "", nil, fmt.Errorf("failed to unmarshal body: %w", err)
	}
	return request.Model, &request, nil
}
