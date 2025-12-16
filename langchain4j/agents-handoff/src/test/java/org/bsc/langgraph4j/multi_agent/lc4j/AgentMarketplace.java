package org.bsc.langgraph4j.multi_agent.lc4j;

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.model.output.structured.Description;
import org.bsc.langgraph4j.GraphStateException;

/**
 * 代理市场
 */
public class AgentMarketplace extends AbstractAgentExecutor<AgentMarketplace.Builder> {

    /**
     * 工具列表
     */
    static class Tools {
        /**
         * 产品
         * @param name 产品名称
         * @param price 产品价格
         * @param currency 产品价格货币
         */
        record Product(
                @Description("the product name") String name,
                @Description("the product price") double price,
                @Description("the product price currency") String currency) {}

        /**
         * 按产品搜索
         * 在市场中搜索特定产品
         * @param product 要搜索的产品名称
         */
        @Tool("search for a specific product in the marketplace")
        Product searchByProduct( @P("the product name to search") String product ) {
            // 具体实现
            return new Product( "X", 1000, "EUR" );
        }

    }
    public static class Builder extends AbstractAgentExecutor.Builder<Builder> {

        public AgentMarketplace build() throws GraphStateException {
            this.name("marketplace")
                    .description("marketplace agent, ask for information about products")
                    // 单个参数
                    .singleParameter("all information request about the products")
                    // 系统消息
                    .systemMessage(SystemMessage.from("""
                                You are the agent that provides the information on the product marketplace.
                            """))
                    // 工具列表
                    .toolFromObject(new Tools());
            return new AgentMarketplace(this);
        }

    }

    public static Builder builder() {
        return new Builder();
    }


    public AgentMarketplace( Builder builder ) throws GraphStateException {
        super( builder );
    }

}
