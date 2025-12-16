package dev.langchain4j.adaptiverag;

import dev.langchain4j.DotEnvConfig;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.chroma.ChromaEmbeddingStore;
import org.bsc.langgraph4j.GraphRepresentation;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.FileInputStream;
import java.time.Duration;
import java.util.List;
import java.util.logging.LogManager;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * 自适应RAG测试
 */
public class AdaptiveRagTest {

    @BeforeAll
    public static void beforeAll() throws Exception {
        FileInputStream configFile = new FileInputStream("logging.properties");
        LogManager.getLogManager().readConfiguration(configFile);

        DotEnvConfig.load();
    }

    String getOpenAiKey() {
        return DotEnvConfig.valueOf("OPENAI_API_KEY")
                .orElseThrow( () -> new IllegalArgumentException("no OPENAI APIKEY provided!"));
    }

    String getTavilyApiKey() {
        return DotEnvConfig.valueOf("TAVILY_API_KEY")
                .orElseThrow( () -> new IllegalArgumentException("no TAVILY APIKEY provided!"));
    }

    @Test
    public void QuestionRewriterTest() {

        // 问题重写器
        String result = (new QuestionRewriter(getOpenAiKey())).apply("agent memory");
        // 聊天记忆在智能体的运作中起什么作用？
        assertEquals("What is the role of memory in an agent's functioning?", result);
    }

    @Test
    public void RetrievalGraderTest() {

        String openApiKey = DotEnvConfig.valueOf("OPENAI_API_KEY")
                .orElseThrow( () -> new IllegalArgumentException("no APIKEY provided!"));

        // 检索评分器
        RetrievalGrader grader = new RetrievalGrader(openApiKey);

        // Chroma的嵌入存储
        ChromaEmbeddingStore chroma = ChromaEmbeddingStore.builder()
//                .apiVersion(V1)
                .baseUrl("http://localhost:8000")
                .collectionName("rag-chroma")
                .logRequests(true)
                .logResponses(true)
                .timeout(Duration.ofMinutes(2))
                .build();

        // OpenAI的嵌入模型
        OpenAiEmbeddingModel embeddingModel = OpenAiEmbeddingModel.builder()
                .apiKey(openApiKey)
                .build();

        String question = "agent memory";
        Embedding queryEmbedding = embeddingModel.embed(question).content();

        // 嵌入搜索
        EmbeddingSearchRequest query = EmbeddingSearchRequest.builder()
                .queryEmbedding( queryEmbedding )
                .maxResults( 1 )
                .minScore( 0.0 )
                .build();
        // 相似性搜索
        EmbeddingSearchResult<TextSegment> relevant = chroma.search( query );

        List<EmbeddingMatch<TextSegment>> matches = relevant.matches();

        assertEquals( 1, matches.size() );

        // 检索评分器
        RetrievalGrader.Score answer =
                grader.apply( new RetrievalGrader.Arguments(question, matches.get(0).embedded().text()));

        assertEquals( "no", answer.binaryScore);
    }

    @Test
    public void WebSearchTest() {

        // 网络搜索工具
        WebSearchTool webSearchTool = new WebSearchTool(getTavilyApiKey());
        List<Content> webSearchResults = webSearchTool.apply("agent memory");

        String webSearchResultsText = webSearchResults.stream().map( content -> content.textSegment().text() )
                .collect(Collectors.joining("\n"));

        assertNotNull( webSearchResultsText );

        System.out.println( webSearchResultsText );
    }

    @Test
    public void questionRouterTest() {

        // 问题路由器
        QuestionRouter qr = new QuestionRouter(getOpenAiKey());

        // 类型路由
        // 什么是股票期权？
        QuestionRouter.Type result = qr.apply( "What are the stock options?");

        assertEquals( QuestionRouter.Type.web_search, result );

        // 代理聊天记忆？
        result = qr.apply( "agent memory?");

        assertEquals( QuestionRouter.Type.vectorstore, result );
    }


    @Test
    public void generationTest() {

        // Chroma商店
        ChromaStore retriever = ChromaStore.of(getOpenAiKey());

        String question = "agent memory";
        EmbeddingSearchResult<TextSegment> relevantDocs = retriever.search(question);

        List<String> docs = relevantDocs.matches().stream()
                                .map( m -> m.embedded().text() )
                                .collect(Collectors.toList());
        // 生成
        Generation qr = new Generation(getOpenAiKey());

        String result = qr.apply( question, docs );

        System.out.println( result );
    }

    @Test
    public void getGraphTest() throws Exception {

        // 自适应RAG
        AdaptiveRag adaptiveRag = new AdaptiveRag(getOpenAiKey(), getTavilyApiKey());

        // 构建状态图
        org.bsc.langgraph4j.StateGraph<AdaptiveRag.State> graph = adaptiveRag.buildGraph();

        // 图表示在代码即图形格式中的表示。
        GraphRepresentation plantUml = graph.getGraph( GraphRepresentation.Type.PLANTUML, "Adaptive RAG" );

        System.out.println( plantUml.content() );

        GraphRepresentation mermaid = graph.getGraph( GraphRepresentation.Type.MERMAID, "Adaptive RAG" );

        System.out.println( mermaid.content() );
    }
}
