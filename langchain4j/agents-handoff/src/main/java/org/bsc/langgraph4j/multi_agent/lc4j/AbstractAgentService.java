package org.bsc.langgraph4j.multi_agent.lc4j;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.service.MemoryId;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.tool.ToolExecutor;
import dev.langchain4j.service.AiServices;

import java.util.Map;

/**
 * 代理服务的抽象基类
 */
public abstract class AbstractAgentService<B extends AbstractAgentService.Builder<B>> extends AbstractAgent<B> {

    public static abstract class Builder<B extends AbstractAgentService.Builder<B>> extends AbstractAgent.Builder<B> {

        /**
         * 代理服务
         */
        final AiServices<Service> delegate;

        public Builder() {
            // 构建代理服务
            this.delegate = AiServices.builder( Service.class );
        }

        public B chatModel(ChatModel model) {
            // 聊天模型
            delegate.chatModel(model);
            return result();
        }

        public B tools(Map.Entry<ToolSpecification, ToolExecutor> entry) {
            // 工具列表
            delegate.tools( entry );
            return result();
        }

        public B toolFromObject( Object objectWithTools ) {
            // 具有工具列表的对象
            delegate.tools(objectWithTools);
            return result();
        }

        public B systemMessage(SystemMessage message) {
            // 系统消息
            delegate.systemMessageProvider( ( param ) -> message.text() );
            return result();
        }
    }

    /**
     * 服务
     */
    interface Service {

        /**
         * 执行
         * @param userMessage 用户消息
         * @param memoryId 聊天记忆ID
         */
        String execute(@UserMessage String userMessage, @MemoryId Object memoryId);
    }

    /**
     * 代理服务
     */
    private final Service agentService;

    public AbstractAgentService( Builder<B> builder )  {
        super( builder );

        // 构建AI代理服务
        agentService = builder.delegate.build();
    }

    @Override
    public String execute(ToolExecutionRequest toolExecutionRequest, Object memoryId) {
        // 执行工具
        return agentService.execute( toolExecutionRequest.arguments(), memoryId );

    }

}
