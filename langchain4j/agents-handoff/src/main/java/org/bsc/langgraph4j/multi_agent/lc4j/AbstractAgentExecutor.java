package org.bsc.langgraph4j.multi_agent.lc4j;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.service.tool.ToolExecutor;
import org.bsc.langgraph4j.CompiledGraph;
import org.bsc.langgraph4j.GraphStateException;
import org.bsc.langgraph4j.agentexecutor.AgentExecutor;

import java.util.Map;

/**
 * 代理执行器的抽象基类
 */
public abstract class AbstractAgentExecutor<B extends AbstractAgentExecutor.Builder<B>> extends AbstractAgent<B> {

    public static abstract class Builder<B extends AbstractAgentExecutor.Builder<B>> extends AbstractAgent.Builder<B> {

        /**
         * 代理执行器的代理
         */
        final AgentExecutor.Builder delegate = AgentExecutor.builder();

        public B chatModel(ChatModel model) {
            // 聊天模型
            delegate.chatModel(model);
            return result();
        }

        public B tool(Map.Entry<ToolSpecification, ToolExecutor> entry) {
            // 工具
            delegate.tool(entry);
            return result();
        }

        public B toolFromObject( Object objectWithTools ) {
            // 具有工具列表的对象
            delegate.toolsFromObject(objectWithTools);
            return result();
        }

        public B systemMessage(SystemMessage message) {
            // 系统消息
            delegate.systemMessage(message);
            return result();
        }
    }

    /**
     * 代理执行器
     * 编译图
     */
    private final CompiledGraph<AgentExecutor.State> agentExecutor;

    public AbstractAgentExecutor( Builder<B> builder ) throws GraphStateException {
        super( builder );

        // 构建代理，并编译状态图
        agentExecutor = builder.delegate.build().compile();
    }

    @Override
    public String execute(ToolExecutionRequest toolExecutionRequest, Object o) {

        // 用户消息
        var userMessage = UserMessage.from( toolExecutionRequest.arguments() );

        // 调用
        var result = agentExecutor.invoke( Map.of( "messages", userMessage ) );

        // 最终回应
        return result.flatMap(AgentExecutor.State::finalResponse).orElseThrow();
    }

}
