# weather.py - 天气查询 MCP 服务器
# 通过美国国家气象局 (NWS) API 提供天气预报和警报查询功能
# 使用 FastMCP 框架，以 stdio 传输方式运行

from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP


# 初始化 FastMCP 服务器实例，名称为 "weather"，日志级别设为 ERROR 以减少输出
mcp = FastMCP("weather", log_level="ERROR")


# NWS API 基础地址和请求头中的 User-Agent 标识
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"


async def make_nws_request(url: str) -> dict[str, Any] | None:
    """向 NWS API 发送请求并返回 JSON 数据，失败时返回 None。"""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"  # NWS API 要求的响应格式
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()  # 如果状态码非 2xx 则抛出异常
            return response.json()
        except Exception:
            return None  # 请求失败时返回 None，由调用方处理


def format_alert(feature: dict) -> str:
    """将单条警报数据格式化为可读字符串，包含事件类型、区域、严重程度等信息。"""
    props = feature["properties"]
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""


# MCP 工具：查询指定美国州的天气警报
@mcp.tool()
async def get_alerts(state: str) -> str:
    """获取指定美国州的天气警报信息。

    Args:
        state: 两位美国州代码（如 CA 表示加利福尼亚，NY 表示纽约）
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    # 将所有警报格式化并用分隔线连接
    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)


# MCP 工具：根据经纬度查询天气预报
@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """根据经纬度获取天气预报。

    Args:
        latitude: 纬度
        longitude: 经度
    """
    # 第一步：通过经纬度获取对应的预报网格端点
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."

    # 第二步：从 points 响应中提取预报 URL，再次请求获取详细预报
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # 第三步：格式化预报数据，只展示最近 5 个时间段的天气信息
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:
        forecast = f"""
{period['name']}:
Temperature: {period['temperature']}°{period['temperatureUnit']}
Wind: {period['windSpeed']} {period['windDirection']}
Forecast: {period['detailedForecast']}
"""
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)


if __name__ == "__main__":
    # 以 stdio 传输方式启动 MCP 服务器，供 Claude 等客户端调用
    mcp.run(transport='stdio')