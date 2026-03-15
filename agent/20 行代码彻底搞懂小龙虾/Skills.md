如果需要获取新闻，使用如下命令，XXX为用户希望搜索的关键词
curl -L -A "Mozilla/5.0" "https://news.google.com/rss/search?q=XXX&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"

if you are asked about current time, call this link to get utc: curl -L -A "Mozilla/5.0" https://timeapi.io/api/v1/time/current/utc