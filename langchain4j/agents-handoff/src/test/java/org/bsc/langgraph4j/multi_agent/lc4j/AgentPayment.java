package org.bsc.langgraph4j.multi_agent.lc4j;

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.model.output.structured.Description;
import org.bsc.langgraph4j.GraphStateException;

/**
 * 代理付款
 */
public class AgentPayment extends AbstractAgentExecutor<AgentPayment.Builder> {

    /**
     * 工具列表
     */
    static class Tools {

        /**
         * 交易
         * @param product 购买的产品名称
         * @param code 代码操作
         */
        record Transaction(
                @Description("the product name bought") String product,
                @Description("code operation") String code
        ) {}

        /**
         * 提交付款
         * 提交购买特定产品的付款
         * @param product 要购买的产品名称
         * @param price 产品价格
         * @param currency 产品价格货币
         * @param iban 国际银行账户号码（IBAN）
         */
        @Tool("submit a payment for purchasing a specific product")
        Transaction submitPayment(
                @P("the product name to buy") String product,
                @P("the product price") double price,
                @P("the product price currency") String currency,
                @P("International Bank Account Number (IBAN)") String iban ) {
            // 具体实现
            return new Transaction( product,"123456789A" );

        }

        /**
         * 检索IBAN信息
         */
        @Tool("retrieve IBAN information")
        String retrieveIBAN() {
            return """
                    GB82WEST12345698765432
                    """;
        }

    }

    public static class Builder extends AbstractAgentExecutor.Builder<AgentPayment.Builder> {

        /**
         * 构建代理付款
         */
        public AgentPayment build() throws GraphStateException {
            // 代理付款
            return new AgentPayment( this.name("payment")
                    .description("payment agent, request purchase and payment transactions")
                    // 单个参数
                    .singleParameter("all information about purchasing to allow the payment")
                    // 系统消息
                    .systemMessage( SystemMessage.from("""
                    You are the agent that provides payment service.
                    """) )
                    // 工具列表
                    .toolFromObject( new Tools() ));
        }

    }

    public static AgentPayment.Builder builder() {
        return new Builder();
    }


    public AgentPayment( Builder builder ) throws GraphStateException {
        super(builder);
    }

}
