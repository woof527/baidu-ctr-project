-- =============================================================================
-- 百度 CTR 项目 — 基础 SQL 分析
-- =============================================================================
-- 数据库：data/interim/baidu_ctr.db
-- 主表：  train_events（训练集曝光点击事件）
--
-- 说明：
--   - 以下均为 SELECT 查询，不会修改数据库
--   - CTR 使用 AVG(click) 计算（click 为 0/1，平均值即点击率）
--   - hour 字段格式为 YYMMDDHH，例如 14102100 表示 2014-10-21 00:00
--
-- 使用方式（示例）：
--   sqlite3 data/interim/baidu_ctr.db < sql/01_basic_analysis.sql
--   或在 sqlite3 交互模式中逐段复制执行
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. 整体 CTR
-- 统计训练集总曝光量、总点击量与整体点击率
-- -----------------------------------------------------------------------------
SELECT
    COUNT(*)   AS impressions,
    SUM(click) AS clicks,
    AVG(click) AS ctr
FROM train_events;


-- -----------------------------------------------------------------------------
-- 2. 按 banner_pos 分析
-- 各广告位编码的曝光、点击与 CTR，按曝光量从高到低排列
-- -----------------------------------------------------------------------------
SELECT
    banner_pos,
    COUNT(*)   AS impressions,
    SUM(click) AS clicks,
    AVG(click) AS ctr
FROM train_events
GROUP BY banner_pos
ORDER BY impressions DESC;


-- -----------------------------------------------------------------------------
-- 3. 按 device_type 分析
-- 各设备类型编码的曝光、点击与 CTR，按曝光量从高到低排列
-- -----------------------------------------------------------------------------
SELECT
    device_type,
    COUNT(*)   AS impressions,
    SUM(click) AS clicks,
    AVG(click) AS ctr
FROM train_events
GROUP BY device_type
ORDER BY impressions DESC;


-- -----------------------------------------------------------------------------
-- 4. 按小时分析
-- 从 hour 字段提取小时（substr 取第 7–8 位，即 HH），按 0–23 点升序
-- -----------------------------------------------------------------------------
SELECT
    substr(hour, 7, 2) AS hour_of_day,
    COUNT(*)             AS impressions,
    SUM(click)           AS clicks,
    AVG(click)           AS ctr
FROM train_events
GROUP BY hour_of_day
ORDER BY hour_of_day ASC;


-- -----------------------------------------------------------------------------
-- 5. site_category Top 15
-- 曝光量最高的 15 个网站类别编码及其 CTR
-- -----------------------------------------------------------------------------
SELECT
    site_category,
    COUNT(*)   AS impressions,
    SUM(click) AS clicks,
    AVG(click) AS ctr
FROM train_events
GROUP BY site_category
ORDER BY impressions DESC
LIMIT 15;


-- -----------------------------------------------------------------------------
-- 6. app_category Top 15
-- 曝光量最高的 15 个应用类别编码及其 CTR
-- -----------------------------------------------------------------------------
SELECT
    app_category,
    COUNT(*)   AS impressions,
    SUM(click) AS clicks,
    AVG(click) AS ctr
FROM train_events
GROUP BY app_category
ORDER BY impressions DESC
LIMIT 15;


-- -----------------------------------------------------------------------------
-- 7. banner_pos × device_type 交叉分析
-- 广告位与设备类型的组合曝光、点击与 CTR，按曝光量从高到低排列
-- -----------------------------------------------------------------------------
SELECT
    banner_pos,
    device_type,
    COUNT(*)   AS impressions,
    SUM(click) AS clicks,
    AVG(click) AS ctr
FROM train_events
GROUP BY banner_pos, device_type
ORDER BY impressions DESC;


-- -----------------------------------------------------------------------------
-- 8. hour_of_day × banner_pos 交叉分析
-- 小时与广告位的组合曝光、点击与 CTR，先按小时、再按广告位升序
-- -----------------------------------------------------------------------------
SELECT
    substr(hour, 7, 2) AS hour_of_day,
    banner_pos,
    COUNT(*)             AS impressions,
    SUM(click)           AS clicks,
    AVG(click)           AS ctr
FROM train_events
GROUP BY hour_of_day, banner_pos
ORDER BY hour_of_day ASC, banner_pos ASC;
