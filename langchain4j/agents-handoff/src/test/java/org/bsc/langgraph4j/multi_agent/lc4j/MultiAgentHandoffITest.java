package org.bsc.langgraph4j.multi_agent.lc4j;

import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Set;

/**
 * 多个智能体交接测试
 */
public class MultiAgentHandoffITest {

    /**
     * AI模型
     */
    enum AiModel {

        OPENAI_GPT_4O_MINI( OpenAiChatModel.builder()
                .apiKey( System.getenv("OPENAI_API_KEY") )
                .modelName( "gpt-4o-mini" )
                .supportedCapabilities(Set.of(Capability.RESPONSE_FORMAT_JSON_SCHEMA))
                .logResponses(true)
                .maxRetries(2)
                .temperature(0.0)
                .build() ),
        OLLAMA_QWEN3_14B( OllamaChatModel.builder()
                .modelName( "qwen3:14b" )
                .baseUrl("http://localhost:11434")
                .supportedCapabilities(Capability.RESPONSE_FORMAT_JSON_SCHEMA)
                .logRequests(true)
                .logResponses(true)
                .maxRetries(2)
                .temperature(0.0)
                .build() ),
        OLLAMA_QWEN2_5_7B( OllamaChatModel.builder()
                .modelName( "qwen2.5:7b" )
                .baseUrl("http://localhost:11434")
                .supportedCapabilities(Capability.RESPONSE_FORMAT_JSON_SCHEMA)
                .logRequests(true)
                .logResponses(true)
                .maxRetries(2)
                .temperature(0.0)
                .build() )
        ;

        /**
         * 聊天模型
         */
        public final ChatModel model;

        AiModel(  ChatModel model ) {
            this.model = model;
        }
    }


    @Test
    public void testHandoff() throws Exception {

        // 代理市场
        var agentMarketplace = AgentMarketplace.builder()
                .chatModel( AiModel.OLLAMA_QWEN2_5_7B.model )
                .build();

        // 代理付款
        var agentPayment = AgentPayment.builder()
                .chatModel( AiModel.OLLAMA_QWEN3_14B.model )
                .build();

        // 代理转接
        var handoffExecutor = AgentHandoff.builder()
                .chatModel(AiModel.OLLAMA_QWEN3_14B.model)
                .agent( agentMarketplace )
                .agent( agentPayment )
                .build()
                .compile()
                ;

        var input = "search for product 'X' and purchase it";

        // 调用
        var result = handoffExecutor.invoke( Map.of( "messages", UserMessage.from(input)));

        System.out.println( result );

    }
}
