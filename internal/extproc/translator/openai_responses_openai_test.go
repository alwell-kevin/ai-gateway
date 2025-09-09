package translator

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
)

func Test_openAIToOpenAITranslatorV1Responses_RequestBody(t *testing.T) {
	translator := NewResponsesOpenAIToOpenAITranslator("v1", "new-model")
	original := []byte(`{"model":"old-model","stream":true}`)
	req := &openai.ResponsesRequest{Model: "old-model", Stream: true}

	header, body, err := translator.RequestBody(original, req, false)
	require.NoError(t, err)
	require.NotNil(t, header)
	require.Len(t, header.SetHeaders, 1)
	require.Equal(t, ":path", header.SetHeaders[0].Header.Key)
	require.Equal(t, "/v1/responses", string(header.SetHeaders[0].Header.RawValue))

	require.NotNil(t, body)
	mutated := body.GetBody()
	require.NotNil(t, mutated)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(mutated, &payload))
	require.Equal(t, "new-model", payload["model"])
	require.Equal(t, true, payload["stream"])
}

func Test_openAIToOpenAITranslatorV1Responses_ResponseBody(t *testing.T) {
	translator := NewResponsesOpenAIToOpenAITranslator("v1", "")
	_, _, err := translator.RequestBody([]byte(`{"model":"foo"}`), &openai.ResponsesRequest{Model: "foo"}, false)
	require.NoError(t, err)

	_, _, tokens, latencyTokens, err := translator.ResponseBody(nil, bytes.NewReader([]byte(`{"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}`)), true)
	require.NoError(t, err)
	require.Equal(t, uint32(1), tokens.InputTokens)
	require.Equal(t, uint32(2), tokens.OutputTokens)
	require.Equal(t, uint32(3), tokens.TotalTokens)
	require.Equal(t, uint32(0), latencyTokens)
}

func Test_openAIToOpenAITranslatorV1Responses_ResponseBodyStreaming(t *testing.T) {
	translator := NewResponsesOpenAIToOpenAITranslator("v1", "")
	_, _, err := translator.RequestBody([]byte(`{"model":"foo","stream":true}`), &openai.ResponsesRequest{Model: "foo", Stream: true}, false)
	require.NoError(t, err)

	deltaPayload := "data: {\"type\":\"response.output_text.delta\"}\n"
	_, _, tokens, latencyTokens, err := translator.ResponseBody(nil, bytes.NewBufferString(deltaPayload), false)
	require.NoError(t, err)
	require.Equal(t, uint32(0), tokens.TotalTokens)
	require.Equal(t, uint32(1), latencyTokens)

	streamPayload := "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":6,\"total_tokens\":11}}}\n\n"
	_, _, tokens, latencyTokens, err = translator.ResponseBody(nil, bytes.NewBufferString(streamPayload), false)
	require.NoError(t, err)
	require.Equal(t, uint32(5), tokens.InputTokens)
	require.Equal(t, uint32(6), tokens.OutputTokens)
	require.Equal(t, uint32(11), tokens.TotalTokens)
	require.Equal(t, uint32(0), latencyTokens)
}

func Test_openAIToOpenAITranslatorV1Responses_ResponseError(t *testing.T) {
	translator := NewResponsesOpenAIToOpenAITranslator("v1", "")
	headers := map[string]string{statusHeaderName: "500", contentTypeHeaderName: "text/plain"}

	headerMutation, bodyMutation, err := translator.ResponseError(headers, bytes.NewBufferString("backend failure"))
	require.NoError(t, err)
	require.NotNil(t, headerMutation)
	require.NotNil(t, bodyMutation)

	body := bodyMutation.GetBody()
	require.NotNil(t, body)

	var parsed map[string]any
	require.NoError(t, json.Unmarshal(body, &parsed))
	require.Equal(t, "error", parsed["type"])
}

func Test_openAIToOpenAITranslatorV1Responses_ResponseBody_NoStreamingUsageUntilBuffered(t *testing.T) {
	translator := NewResponsesOpenAIToOpenAITranslator("v1", "")
	_, _, err := translator.RequestBody([]byte(`{"model":"foo","stream":true}`), &openai.ResponsesRequest{Model: "foo", Stream: true}, false)
	require.NoError(t, err)

	// First chunk does not contain usage yet.
	_, _, tokens, latencyTokens, err := translator.ResponseBody(nil, bytes.NewBufferString("data: {\"type\":\"response.created\"}\n"), false)
	require.NoError(t, err)
	require.Equal(t, uint32(0), tokens.TotalTokens)
	require.Equal(t, uint32(0), latencyTokens)

	// Usage chunk should now be extracted.
	_, _, tokens, latencyTokens, err = translator.ResponseBody(nil, bytes.NewBufferString("data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":2,\"output_tokens\":4,\"total_tokens\":6}}}\n"), true)
	require.NoError(t, err)
	require.Equal(t, uint32(2), tokens.InputTokens)
	require.Equal(t, uint32(4), tokens.OutputTokens)
	require.Equal(t, uint32(6), tokens.TotalTokens)
	require.Equal(t, uint32(0), latencyTokens)
}

func Test_openAIToOpenAITranslatorV1Responses_ResponseBody_ErrorWhenInvalidJSON(t *testing.T) {
	translator := NewResponsesOpenAIToOpenAITranslator("v1", "")
	_, _, err := translator.RequestBody([]byte(`{"model":"foo"}`), &openai.ResponsesRequest{Model: "foo"}, false)
	require.NoError(t, err)

	_, _, _, _, err = translator.ResponseBody(nil, bytes.NewBufferString("{invalid"), true)
	require.Error(t, err)
}
