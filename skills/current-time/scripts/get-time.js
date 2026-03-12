// 获取当前系统时间并输出详细信息
const now = new Date();

const pad = (n) => String(n).padStart(2, '0');

const year = now.getFullYear();
const month = pad(now.getMonth() + 1);
const day = pad(now.getDate());
const hours = pad(now.getHours());
const minutes = pad(now.getMinutes());
const seconds = pad(now.getSeconds());
const ms = String(now.getMilliseconds()).padStart(3, '0');

const weekdays = ['日', '一', '二', '三', '四', '五', '六'];
const weekday = weekdays[now.getDay()];

const tzOffsetMin = -now.getTimezoneOffset();
const tzSign = tzOffsetMin >= 0 ? '+' : '-';
const tzHours = pad(Math.floor(Math.abs(tzOffsetMin) / 60));
const tzMins = pad(Math.abs(tzOffsetMin) % 60);
const timezone = `UTC${tzSign}${tzHours}:${tzMins}`;

const result = {
  datetime: `${year}-${month}-${day} ${hours}:${minutes}:${seconds}.${ms}`,
  date: `${year}-${month}-${day}`,
  time: `${hours}:${minutes}:${seconds}`,
  year,
  month: parseInt(month),
  day: parseInt(day),
  weekday: `星期${weekday}`,
  hours: parseInt(hours),
  minutes: parseInt(minutes),
  seconds: parseInt(seconds),
  timestamp: now.getTime(),
  iso: now.toISOString(),
  timezone,
};

console.log(JSON.stringify(result, null, 2));
