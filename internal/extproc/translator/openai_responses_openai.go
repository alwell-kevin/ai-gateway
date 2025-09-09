package translator

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"path"
	"strconv"
	"strings"

	corev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extprocv3 "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/tidwall/sjson"

	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
)

// NewResponsesOpenAIToOpenAITranslator returns a translator for the
// /v1/responses endpoint when the backend is also OpenAI compatible.
func NewResponsesOpenAIToOpenAITranslator(apiVersion string, modelNameOverride string) OpenAIResponsesTranslator {
	return &openAIToOpenAITranslatorV1Responses{
		modelNameOverride: modelNameOverride,
		path:              path.Join("/", apiVersion, "responses"),
	}
}

type openAIToOpenAITranslatorV1Responses struct {
	modelNameOverride string
	path              string
	stream            bool
	buffered          []byte
	bufferingDone     bool
}

func (o *openAIToOpenAITranslatorV1Responses) RequestBody(original []byte, req *openai.ResponsesRequest, forceBodyMutation bool) (
	headerMutation *extprocv3.HeaderMutation,
	bodyMutation *extprocv3.BodyMutation,
	err error,
) {
	if req != nil {
		o.stream = req.Stream
	}

	var newBody []byte
	if o.modelNameOverride != "" {
		newBody, err = sjson.SetBytesOptions(original, "model", o.modelNameOverride, SJSONOptions)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to set model name: %w", err)
		}
	}

	headerMutation = &extprocv3.HeaderMutation{
		SetHeaders: []*corev3.HeaderValueOption{{
			Header: &corev3.HeaderValue{
				Key:      ":path",
				RawValue: []byte(o.path),
			},
		}},
	}

	if forceBodyMutation && len(newBody) == 0 {
		newBody = original
	}

	if len(newBody) > 0 {
		bodyMutation = &extprocv3.BodyMutation{
			Mutation: &extprocv3.BodyMutation_Body{Body: newBody},
		}
		headerMutation.SetHeaders = append(headerMutation.SetHeaders, &corev3.HeaderValueOption{
			Header: &corev3.HeaderValue{
				Key:      "content-length",
				RawValue: []byte(strconv.Itoa(len(newBody))),
			},
		})
	}

	return
}

func (o *openAIToOpenAITranslatorV1Responses) ResponseHeaders(map[string]string) (*extprocv3.HeaderMutation, error) {
	return nil, nil
}

func (o *openAIToOpenAITranslatorV1Responses) ResponseBody(_ map[string]string, body io.Reader, _ bool) (
	headerMutation *extprocv3.HeaderMutation,
	bodyMutation *extprocv3.BodyMutation,
	tokenUsage LLMTokenUsage,
	latencyTokens uint32,
	err error,
) {
	if o.stream {
		if o.bufferingDone {
			return nil, nil, tokenUsage, 0, nil
		}
		buf, err := io.ReadAll(body)
		if err != nil {
			return nil, nil, tokenUsage, 0, fmt.Errorf("failed to read body: %w", err)
		}
		o.buffered = append(o.buffered, buf...)
		tokenUsage, latencyTokens = o.extractUsageFromBufferEvent()
		return nil, nil, tokenUsage, latencyTokens, nil
	}

	resp := &openai.ResponsesResponse{}
	if err := json.NewDecoder(body).Decode(resp); err != nil {
		return nil, nil, tokenUsage, 0, fmt.Errorf("failed to unmarshal body: %w", err)
	}
	tokenUsage = convertResponsesUsage(resp.Usage)
	return nil, nil, tokenUsage, 0, nil
}

func (o *openAIToOpenAITranslatorV1Responses) ResponseError(respHeaders map[string]string, body io.Reader) (
	headerMutation *extprocv3.HeaderMutation,
	bodyMutation *extprocv3.BodyMutation,
	err error,
) {
	statusCode := respHeaders[statusHeaderName]
	if v, ok := respHeaders[contentTypeHeaderName]; ok && v != jsonContentType {
		var openaiError openai.Error
		buf, err := io.ReadAll(body)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to read error body: %w", err)
		}
		openaiError = openai.Error{
			Type: "error",
			Error: openai.ErrorType{
				Type:    openAIBackendError,
				Message: string(buf),
				Code:    &statusCode,
			},
		}
		mut := &extprocv3.BodyMutation_Body{}
		mut.Body, err = json.Marshal(openaiError)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to marshal error body: %w", err)
		}
		headerMutation = &extprocv3.HeaderMutation{}
		setContentLength(headerMutation, mut.Body)
		return headerMutation, &extprocv3.BodyMutation{Mutation: mut}, nil
	}
	return nil, nil, nil
}

func (o *openAIToOpenAITranslatorV1Responses) extractUsageFromBufferEvent() (tokenUsage LLMTokenUsage, latencyTokens uint32) {
	for {
		idx := bytes.IndexByte(o.buffered, '\n')
		if idx == -1 {
			return tokenUsage, latencyTokens
		}
		line := o.buffered[:idx]
		o.buffered = o.buffered[idx+1:]
		if !bytes.HasPrefix(line, dataPrefix) {
			continue
		}
		payload := bytes.TrimPrefix(line, dataPrefix)
		payload = bytes.TrimSpace(payload)
		if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
			continue
		}
		event := &openai.ResponsesStreamEvent{}
		if err := json.Unmarshal(payload, event); err != nil {
			continue
		}
		if event.Type == "response.completed" {
			if event.Response == nil {
				continue
			}
			tokenUsage = convertResponsesUsage(event.Response.Usage)
			o.buffered = nil
			o.bufferingDone = true
			return tokenUsage, latencyTokens
		}
		if strings.HasSuffix(event.Type, ".delta") {
			latencyTokens++
		}
	}
}

func convertResponsesUsage(usage openai.ResponsesUsage) LLMTokenUsage {
	return LLMTokenUsage{
		InputTokens:  uint32(usage.InputTokens),
		OutputTokens: uint32(usage.OutputTokens),
		TotalTokens:  uint32(usage.TotalTokens),
	}
}
