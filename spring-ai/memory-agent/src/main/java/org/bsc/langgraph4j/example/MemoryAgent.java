package org.bsc.langgraph4j.example;

import org.bsc.langgraph4j.CompileConfig;
import org.bsc.langgraph4j.CompiledGraph;
import org.bsc.langgraph4j.GraphStateException;
import org.bsc.langgraph4j.StateGraph;
import org.bsc.langgraph4j.checkpoint.BaseCheckpointSaver;
import org.bsc.langgraph4j.checkpoint.MemorySaver;
import org.bsc.langgraph4j.spring.ai.agentexecutor.AgentExecutor;
import org.springframework.ai.chat.model.ChatModel;

/**
 * 聊天记忆代理
 * @author lambochen
 */
public class MemoryAgent {

    /**
     * 状态图
     * 代理执行器状态
     */
    private final StateGraph<AgentExecutor.State> graph;
    /**
     * 已编译图的工作流程
     * 代理执行器状态
     */
    private final CompiledGraph<AgentExecutor.State> workflow;

    /**
     * @param llm 聊天模型
     */
    public MemoryAgent(ChatModel llm) throws GraphStateException {
        // 聊天模型
        // 内存的保存器
        this(llm, new MemorySaver());
    }

    /**
     * @param llm 聊天模型
     * @param memorySaver 基础检查点的保存器
     */
    public MemoryAgent(ChatModel llm, BaseCheckpointSaver memorySaver) throws GraphStateException {
        // 构建状态图
        this.graph = AgentExecutor.builder()
                .chatModel(llm)
                .build();

        // 构建工作流程
        this.workflow = graph.compile(
                CompileConfig.builder()
                        .checkpointSaver(memorySaver)
                        .build()
        );
    }

    public CompiledGraph<AgentExecutor.State> workflow() {
        return workflow;
    }

    public CompiledGraph<AgentExecutor.State> newWorkflow(CompileConfig config) throws GraphStateException {
        return graph.compile(config);
    }

    public CompiledGraph<AgentExecutor.State> newWorkflow(MemorySaver memory) throws GraphStateException {
        return graph.compile(
                CompileConfig.builder().checkpointSaver(memory).build()
        );
    }
}
