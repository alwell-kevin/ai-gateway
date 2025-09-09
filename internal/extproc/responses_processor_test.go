package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"testing"

	corev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extprocv3 "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/envoyproxy/ai-gateway/filterapi"
	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/extproc/translator"
	tracing "github.com/envoyproxy/ai-gateway/internal/tracing/api"
)

func TestResponses_Schema(t *testing.T) {
	cfg := &processorConfig{}
	factory := ResponsesProcessorFactory(nil)

	t.Run("router", func(t *testing.T) {
		proc, err := factory(cfg, nil, slog.Default(), tracing.NoopTracing{}, false)
		require.NoError(t, err)
		require.IsType(t, &responsesProcessorRouterFilter{}, proc)
	})

	t.Run("upstream", func(t *testing.T) {
		proc, err := factory(cfg, nil, slog.Default(), tracing.NoopTracing{}, true)
		require.NoError(t, err)
		require.IsType(t, &responsesProcessorUpstreamFilter{}, proc)
	})
}

func Test_responsesProcessorUpstreamFilter_SelectTranslator(t *testing.T) {
	p := &responsesProcessorUpstreamFilter{}

	require.ErrorContains(t, p.selectTranslator(filterapi.VersionedAPISchema{Name: "Unknown"}), "unsupported API schema")

	require.NoError(t, p.selectTranslator(filterapi.VersionedAPISchema{Name: filterapi.APISchemaOpenAI}))
	require.NotNil(t, p.translator)
}

func Test_responsesProcessorRouterFilter_ProcessRequestBody(t *testing.T) {
	t.Run("invalid json", func(t *testing.T) {
		p := &responsesProcessorRouterFilter{}
		_, err := p.ProcessRequestBody(t.Context(), &extprocv3.HttpBody{Body: []byte("foo")})
		require.ErrorContains(t, err, "failed to parse request body")
	})

	t.Run("success", func(t *testing.T) {
		headers := map[string]string{":path": "/v1/responses"}
		p := &responsesProcessorRouterFilter{
			config:         &processorConfig{modelNameHeaderKey: "x-model"},
			requestHeaders: headers,
			logger:         slog.Default(),
		}
		resp, err := p.ProcessRequestBody(t.Context(), &extprocv3.HttpBody{Body: responsesBodyFromModel(t, "some-model")})
		require.NoError(t, err)
		require.NotNil(t, resp)

		rb, ok := resp.Response.(*extprocv3.ProcessingResponse_RequestBody)
		require.True(t, ok)
		setHeaders := rb.RequestBody.GetResponse().GetHeaderMutation().SetHeaders
		require.Len(t, setHeaders, 2)
		require.Equal(t, "x-model", setHeaders[0].Header.Key)
		require.Equal(t, "some-model", string(setHeaders[0].Header.RawValue))
		require.Equal(t, originalPathHeader, setHeaders[1].Header.Key)
		require.Equal(t, "/v1/responses", string(setHeaders[1].Header.RawValue))
	})
}

func Test_responsesProcessorUpstreamFilter_ProcessResponseHeaders(t *testing.T) {
	t.Run("translator error", func(t *testing.T) {
		mt := &mockResponsesTranslator{t: t, expHeaders: make(map[string]string), retErr: errors.New("translator error")}
		mm := &mockChatCompletionMetrics{}
		p := &responsesProcessorUpstreamFilter{
			translator: mt,
			metrics:    mm,
		}
		_, err := p.ProcessResponseHeaders(context.Background(), &corev3.HeaderMap{})
		require.ErrorContains(t, err, "translator error")
		mm.RequireRequestFailure(t)
	})

	t.Run("success", func(t *testing.T) {
		headers := &corev3.HeaderMap{Headers: []*corev3.HeaderValue{{Key: "foo", Value: "bar"}}}
		mt := &mockResponsesTranslator{t: t, expHeaders: map[string]string{"foo": "bar"}}
		mm := &mockChatCompletionMetrics{}
		p := &responsesProcessorUpstreamFilter{translator: mt, metrics: mm, responseHeaders: map[string]string{}, requestHeaders: map[string]string{}}

		resp, err := p.ProcessResponseHeaders(context.Background(), headers)
		require.NoError(t, err)
		require.NotNil(t, resp)
		mm.RequireRequestNotCompleted(t)
	})
}

func Test_responsesProcessorUpstreamFilter_ProcessResponseBody(t *testing.T) {
	t.Run("translator error response", func(t *testing.T) {
		mt := &mockResponsesTranslator{t: t, retHeaderMutation: &extprocv3.HeaderMutation{}, retBodyMutation: &extprocv3.BodyMutation{}, expResponseBody: &extprocv3.HttpBody{Body: []byte("oops")}}
		mm := &mockChatCompletionMetrics{}
		p := &responsesProcessorUpstreamFilter{
			translator:      mt,
			metrics:         mm,
			responseHeaders: map[string]string{":status": "500"},
		}
		resp, err := p.ProcessResponseBody(context.Background(), &extprocv3.HttpBody{Body: []byte("oops"), EndOfStream: true})
		require.NoError(t, err)
		require.NotNil(t, resp)
		mm.RequireRequestSuccess(t)
	})

	t.Run("success", func(t *testing.T) {
		mt := &mockResponsesTranslator{
			t:               t,
			retUsedToken:    translator.LLMTokenUsage{InputTokens: 2, OutputTokens: 3, TotalTokens: 5},
			expResponseBody: &extprocv3.HttpBody{Body: []byte("{}")},
		}
		mm := &mockChatCompletionMetrics{}
		p := &responsesProcessorUpstreamFilter{
			translator:      mt,
			metrics:         mm,
			responseHeaders: map[string]string{":status": "200"},
			requestHeaders:  map[string]string{},
			config:          &processorConfig{},
		}
		resp, err := p.ProcessResponseBody(context.Background(), &extprocv3.HttpBody{Body: []byte("{}"), EndOfStream: true})
		require.NoError(t, err)
		require.NotNil(t, resp)
		mm.RequireTokensRecorded(t, 1)
		mm.RequireRequestSuccess(t)
	})

	t.Run("streaming latency metadata", func(t *testing.T) {
		mt := &mockResponsesTranslator{
			t:                t,
			retUsedToken:     translator.LLMTokenUsage{InputTokens: 1, OutputTokens: 2, TotalTokens: 3},
			retLatencyTokens: 1,
			expResponseBody:  &extprocv3.HttpBody{Body: []byte("{}")},
		}
		mm := &mockChatCompletionMetrics{}
		p := &responsesProcessorUpstreamFilter{
			translator:      mt,
			metrics:         mm,
			responseHeaders: map[string]string{":status": "200"},
			requestHeaders:  map[string]string{"model": "foo"},
			config: &processorConfig{
				metadataNamespace:  "ai_gateway_llm_ns",
				modelNameHeaderKey: "model",
				requestCosts: []processorConfigRequestCost{{
					LLMRequestCost: &filterapi.LLMRequestCost{
						Type:        filterapi.LLMRequestCostTypeOutputToken,
						MetadataKey: "output_token_usage",
					},
				}},
			},
			backendName:       "backend",
			modelNameOverride: "model-override",
			stream:            true,
		}
		resp, err := p.ProcessResponseBody(context.Background(), &extprocv3.HttpBody{Body: []byte("{}"), EndOfStream: true})
		require.NoError(t, err)
		require.NotNil(t, resp)
		mm.RequireTokensRecorded(t, 1)
		mm.RequireRequestSuccess(t)

		metadata := resp.DynamicMetadata.Fields["ai_gateway_llm_ns"].GetStructValue()
		require.NotNil(t, metadata)
		require.Equal(t, float64(2), metadata.Fields["output_token_usage"].GetNumberValue())
		require.Equal(t, float64(1000), metadata.Fields["token_latency_ttft"].GetNumberValue())
		require.Equal(t, float64(500), metadata.Fields["token_latency_itl"].GetNumberValue())
		require.Equal(t, "model-override", metadata.Fields["model_name_override"].GetStringValue())
		require.Equal(t, "backend", metadata.Fields["backend_name"].GetStringValue())
	})
}

func responsesBodyFromModel(t *testing.T, model string) []byte {
	t.Helper()
	body := map[string]any{
		"model": model,
		"input": "hello",
	}
	raw, err := json.Marshal(body)
	require.NoError(t, err)
	return raw
}

func Test_parseOpenAIResponsesBody(t *testing.T) {
	model, req, err := parseOpenAIResponsesBody(&extprocv3.HttpBody{Body: responsesBodyFromModel(t, "foo")})
	require.NoError(t, err)
	require.Equal(t, "foo", model)
	require.Equal(t, "foo", req.Model)
	require.False(t, req.Stream)

	// Accept non-boolean stream values and treat them as enabling streaming.
	raw, err := json.Marshal(map[string]any{"model": "bar", "input": "hi", "stream": map[string]any{"type": "sse"}})
	require.NoError(t, err)
	_, req, err = parseOpenAIResponsesBody(&extprocv3.HttpBody{Body: raw})
	require.NoError(t, err)
	require.True(t, req.Stream)

	_, _, err = parseOpenAIResponsesBody(&extprocv3.HttpBody{Body: []byte("not json")})
	require.ErrorContains(t, err, "failed to unmarshal body")
}
