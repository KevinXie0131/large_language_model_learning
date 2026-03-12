from mcp.server.fastmcp import FastMCP

mcp = FastMCP("xuanyuan-universe")


@mcp.tool("get_article_list", description="获取文章列表")
def get_article_list():
    return [
        "我开发了一个抓包软件！",
        "程序员赛道太卷，逆向工程师怎么样？",
        "解密HTTPS加密数据的工具"
    ]


@mcp.tool("get_relate_video_list", description="获取某个主题方面的热门视频列表")
def get_relate_video_list(topic):
    if topic == "编程":
        return [
            {
                "title": "【趣话Redis】我是Redis🟨MySQL大哥被我坑惨了🤯",
                "url": "https://www.bilibili.com/video/BV1Fd4y1T7pD/",
                "view_count": "28万"
            },
            {
                "title": "公司聊天软件禁止截图，怎么破？",
                "url": "https://www.bilibili.com/video/BV1tGHLeiEuA/",
                "view_count": "36.8万"
            }
        ]

    elif topic == "AI":
        return [
            {
                "title": "IDA+DeepSeek🤖AI自动做逆向分析🧐,抢饭碗的来了?🤯",
                "url": "https://www.bilibili.com/video/BV1qdRmYGEsh/",
                "view_count": "4.2万"
            },
            {
                "title": "有了这个神器，再也不怕屎山代码了！",
                "url": "https://www.bilibili.com/video/BV16p4y1m7HD/",
                "view_count": "6.1万"
            }
        ]


if __name__ == "__main__":
    mcp.run(transport='stdio')