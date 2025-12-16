package dev.langchain4j.adaptiverag;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;
import java.time.Duration;
import java.util.List;
import java.util.function.BiFunction;

/**
 * 生成
 * 这个类提供了一个服务，用于使用自然语言处理生成问题的答案。
 * This class provides a service for generating answers to questions using natural language processing.
 * It implements the {@link BiFunction} interface to process question and context inputs and produce an answer.
 */
public record Generation(String openApiKey)  implements BiFunction<String, List<String>, String> {

    /**
     * 服务实现的接口，提供基于给定上下文回答问题的能力。
     * Interface for service implementation that provides capabilities to answer questions based on given context.
     */
    public interface Service {

        /**
         * 根据提供的上下文检索并回答问题。
         * Retrieves and answers a question based on the provided context.
         * <p>
         * The method accepts a question and a list of context strings. It processes these inputs 
         * to formulate an answer. If the answer cannot be determined with certainty, the response 
         * will state that an answer is not known. The provided answer is kept concise and limited 
         * to three sentences.
         *
         * @param question The user's question for which an answer is required.
         * @param context A list of string values representing contextual information useful in answering the question.
         * @return A string containing the answer to the question based on the available context.
         */
        @UserMessage("You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n" +
                "Question: {{question}} \n" +
                "Context: {{context}} \n" +
                "Answer:")
        String invoke(@V("question") String question, @V("context") List<String> context );
    }

    /**
     * Apply a text-based query to an AI model and retrieve the model's response.
     *
     * @param question The text query to be processed by the AI model.
     * @param context A list of strings providing additional context to the query.
     * @return The response from the AI model as a string.
     */
    public String apply( String question, List<String> context) {

        // 聊天模型
        ChatModel chatLanguageModel = OpenAiChatModel.builder()
                .apiKey( openApiKey )
                .modelName( "gpt-3.5-turbo" )
                // 超时
                .timeout(Duration.ofMinutes(2))
                .logRequests(true)
                .logResponses(true)
                .maxRetries(2)
                .temperature(0.0)
                .maxTokens(2000)
                .build();

        // 服务
        Service service = AiServices.create(Service.class, chatLanguageModel);

        // 调用-问答
        return service.invoke( question, context ); // service
    }

}