package org.bsc.langgraph4j.multi_agent.lc4j;

import dev.langchain4j.model.chat.ChatModel;
import org.bsc.langgraph4j.GraphStateException;
import org.bsc.langgraph4j.StateGraph;
import org.bsc.langgraph4j.agentexecutor.AgentExecutor;

import java.util.Objects;

/**
 * 代理转接
 */
public interface AgentHandoff {

   class Builder {
       /**
        * 代理
        */
        final AgentExecutor.Builder delegate = AgentExecutor.builder();

        public Builder chatModel(ChatModel model) {
            // 聊天模型
            delegate.chatModel(model);
            return this;
        }

        public <B extends AbstractAgent.Builder<B>> Builder agent(AbstractAgent<B> agent) {
            // 代理的工具
            delegate.tool( Objects.requireNonNull(agent, "agent cannot be null").asTool() );
            return this;
        }

        public StateGraph<AgentExecutor.State> build() throws GraphStateException {
            // 代理的状态图
            return delegate.build();
        }

    }

    static Builder builder() {
        return new Builder();
    }

}
